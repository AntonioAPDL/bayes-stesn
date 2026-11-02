"""Render the implementation report from immutable saved results only.

No fitting, model selection, interval recalibration or score redefinition occurs
here. Run render_figures.py first. The HTML is self-contained, including media.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR','/tmp/bstesn-mpl')
import numpy as np
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether, Preformatted,
)

ROOT=Path(__file__).resolve().parents[1]
NAVY='#14283b';TEAL='#007f88';CORAL='#e76750';GOLD='#b58b38'
INK='#263746';MUTED='#596d7a';PALE='#eef4f5';LINE='#d8e2e5'
MODEL_LABELS={'ridge_short':'Ridge / recent history','ridge_long':'Ridge / more history',
              'ridge_expressive_long':'Ridge / larger reservoir','group_rhs_expressive_long':'RHS / larger reservoir',
              'group_gaussian_expressive_long':'Grouped Gaussian / larger reservoir'}
SCENARIO_LABELS={'matched_gaussian':'Isotropic Gaussian data','anisotropic_gaussian':'Anisotropic Gaussian data'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def test_count(release):
    # JUnit's suite count includes subtests; counting testcase tags would
    # undercount unittest subtests in current pytest releases.
    import xml.etree.ElementTree as ET
    count=release['installed_tests']['tests']
    path=ROOT/'results'/'validation'/'installed_pytest.xml'
    if path.exists():
        root=ET.parse(path).getroot()
        suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
        if sum(int(s.attrib.get('tests',0)) for s in suites)!=count:
            raise ValueError('Release manifest and installed JUnit count disagree')
    return count


def evidence(study):
    summary=load(study/'summary.json');protocol=load(study/'protocol.json')
    if summary['protocol_sha256']!=protocol['protocol_sha256']:
        raise ValueError('Study summary and protocol identities disagree')
    cases=[]
    for item in summary['case_results']:
        case=load(study/item['path'])
        if case['protocol_sha256']!=protocol['protocol_sha256']:
            raise ValueError('Case has a different frozen protocol')
        cases.append(case)
    illustration=study/'final'/'matched_gaussian'/'replication_000'/'illustration.npz'
    with np.load(illustration,allow_pickle=False) as arrays:
        if str(arrays['protocol_sha256'])!=protocol['protocol_sha256']:
            raise ValueError('Illustration belongs to another study')
        picture={k:arrays[k].copy() for k in arrays.files}
    parity=load(ROOT/'results'/'validation'/'v3_full_parity.json')
    sbc=load(ROOT/'results'/'statistical_validation.json')
    spatial=load(ROOT/'results'/'spatial_approximation_validation.json')
    release=load(ROOT/'results'/'release_manifest.json')
    return summary,protocol,cases,picture,parity,sbc,spatial,release


def comparison_rows(summary,domain='input'):
    rows=[]
    for scenario,result in summary['scenario_results'].items():
        for model,means in result['model_means'].items():
            failed=len(result['failed_replication_ids'][model])
            total=summary['replications_per_scenario']
            scores=means[domain]['observation'] if means else None
            rows.append([SCENARIO_LABELS.get(scenario,scenario),MODEL_LABELS.get(model,model),
                str(total-failed)+'/'+str(total),
                f"{scores['rmse']:.4f}" if scores else 'No eligible fit',
                f"{scores['crps']:.4f}" if scores else '-',
                f"{scores['intervals']['0.9']['coverage']:.3f}" if scores else '-'])
    return rows


def paired_rows(summary,domain='input'):
    rows=[]
    labels={'ridge_long_minus_ridge_short':'More history',
            'ridge_expressive_long_minus_ridge_long':'Larger reservoir',
            'group_rhs_expressive_long_minus_ridge_expressive_long':'RHS instead of ridge',
            'group_gaussian_expressive_long_minus_ridge_expressive_long':'Group scales instead of ridge',
            'validation_selected_minus_ridge_short':'Validation-selected model'}
    for scenario,result in summary['scenario_results'].items():
        for name,entry in result['paired_comparisons'].items():
            value=entry['domains'][domain]['crps']
            if value is None:
                rows.append([SCENARIO_LABELS.get(scenario,scenario),labels.get(name,name),'-','Insufficient eligible pairs'])
            else:
                lo,hi=value['ci']
                rows.append([SCENARIO_LABELS.get(scenario,scenario),labels.get(name,name),str(value['replications']),
                    f"{value['difference']:+.6f} [{lo:+.6f}, {hi:+.6f}]"])
    return rows


def findings(summary):
    paragraphs=[]
    for scenario,result in summary['scenario_results'].items():
        entry=result['paired_comparisons'].get('validation_selected_minus_ridge_short')
        value=entry['domains']['input']['crps'] if entry else None
        if value is None:
            paragraphs.append(f"{SCENARIO_LABELS.get(scenario,scenario)}: no eligible paired comparison of the validation-selection policy with the recent-history baseline.")
            continue
        lo,hi=value['ci']
        baseline=result['model_means']['ridge_short']['input']['observation']['crps']
        message=(f"{SCENARIO_LABELS.get(scenario,scenario)}: the historical validation-selection policy changes input-site CRPS by {value['difference']:+.4f} "
                 f"(paired 95% bootstrap interval [{lo:+.4f}, {hi:+.4f}], {value['replications']} independent replications).")
        if value['replications']==summary['replications_per_scenario'] and baseline>0:
            message+=f" This is a {-100*value['difference']/baseline:.1f}% reduction relative to the recent-history ridge baseline."
        withheld=entry['domains']['heldout']['crps']
        if withheld:
            message+=f" At withheld sites, the paired CRPS difference is {withheld['difference']:+.4f}."
        paragraphs.append(message)
    return paragraphs


def init_styles():
    folder=Path('/usr/share/fonts/truetype/dejavu')
    for name,filename in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf'),
                          ('DejaVuMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(folder/filename)))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyClean',fontName='DejaVu',fontSize=9.6,leading=14.6,textColor=colors.HexColor(INK),spaceAfter=9))
    styles.add(ParagraphStyle(name='HeadingClean',fontName='DejaVu-Bold',fontSize=23,leading=29,textColor=colors.HexColor(NAVY),spaceAfter=15))
    styles.add(ParagraphStyle(name='SubClean',fontName='DejaVu-Bold',fontSize=12,leading=16,textColor=colors.HexColor(TEAL),spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle(name='LabelClean',fontName='DejaVu-Bold',fontSize=8,leading=11,textColor=colors.HexColor(TEAL),spaceAfter=9))
    styles.add(ParagraphStyle(name='CaptionClean',fontName='DejaVu',fontSize=8,leading=11.5,textColor=colors.HexColor(MUTED),spaceAfter=10))
    styles.add(ParagraphStyle(name='TableClean',fontName='DejaVu',fontSize=8,leading=11,textColor=colors.HexColor(INK)))
    styles.add(ParagraphStyle(name='TableHeadClean',fontName='DejaVu-Bold',fontSize=8,leading=11,textColor=colors.white))
    styles.add(ParagraphStyle(name='CoverClean',fontName='DejaVu-Bold',fontSize=42,leading=48,textColor=colors.HexColor(NAVY),spaceAfter=18))
    styles.add(ParagraphStyle(name='CodeClean',fontName='DejaVuMono',fontSize=7.5,leading=11,textColor=colors.HexColor(INK),backColor=colors.HexColor(PALE),borderPadding=12,spaceAfter=10))
    return styles


def build_pdf(output,study,figures,values):
    summary,protocol,cases,picture,parity,sbc,spatial,release=values
    config=protocol['config'];styles=init_styles();story=[]
    width=A4[0]-34*mm
    def p(text,style='BodyClean'):
        story.append(Paragraph(text,styles[style]))
    def section(number,title,lead=None):
        if story:
            story.append(PageBreak())
        p(f'{number:02d} / IMPLEMENTATION &amp; EVIDENCE','LabelClean')
        p(title,'HeadingClean')
        if lead:
            p(lead)
    def table(headers,rows,widths=None):
        data=[[Paragraph(escape(str(c)),styles['TableHeadClean']) for c in headers]]
        data += [[Paragraph(escape(str(c)),styles['TableClean']) for c in row] for row in rows]
        t=Table(data,colWidths=[width*w for w in widths] if widths else None,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(PALE)]),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(TEAL)),
            ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor(LINE))]))
        story.append(t);story.append(Spacer(1,10))
    def fig(filename,maxheight=355,caption=None):
        path=figures/filename
        with PILImage.open(path) as im:
            iw,ih=im.size
        factor=min(width/iw,maxheight/ih)
        story.append(Image(str(path),width=iw*factor,height=ih*factor,hAlign='CENTER'))
        story.append(Spacer(1,7))
        if caption:
            p(caption,'CaptionClean')
    def equation(text):
        text=re.sub(r"\\mathbf\s+([A-Za-z])",r"\\mathbf{\1}",text)
        text=re.sub(r"\\mathcal\s+([A-Za-z])",r"\\mathcal{\1}",text)
        text=re.sub(r"\\boldsymbol(\\[A-Za-z]+)",r"\\boldsymbol{\1}",text)
        buffer=io.BytesIO()
        math_to_image(text,buffer,prop=FontProperties(size=15),dpi=260,format='png',color=NAVY)
        buffer.seek(0)
        with PILImage.open(buffer) as im:
            iw,ih=im.size
        buffer.seek(0)
        factor=min(width/iw,72/260)
        story.append(Image(buffer,width=iw*factor,height=ih*factor,hAlign='LEFT'))
        story.append(Spacer(1,10))

    p('BAYES-STESN / RESEARCH SOFTWARE RELEASE','LabelClean')
    story.append(Spacer(1,20))
    p('Learning<br/>evolving fields.','CoverClean')
    p('Bayesian spatiotemporal echo state networks','SubClean')
    p('Implementation, mathematical verification, controlled forecasting experiments and reproducible visual evidence.')
    story.append(Spacer(1,10))
    fig('forecast_montage.png',maxheight=305)
    table(['MODEL EXTENSIONS','NUMERICAL / API CHECKS','PAIRED SIMULATIONS'],
          [['23 registered IDs',str(test_count(release)),f"{summary['replications_per_scenario']} per scenario / 2 scenarios"]],[.34,.33,.33])
    p('Prepared 02 November 2026. Research release 0.1.0. All displayed results come from saved artifacts. The illustrations use the prespecified replication and its validation-selected model.','CaptionClean')

    section(1,'What was delivered','A connected implementation of the mathematical blueprint, with explicit distinctions between numerically verified paths, experimental model families and reference spatial approximations.')
    for text in findings(summary):
        p(escape(text))
    p('The original full-size model is preserved','SubClean')
    p(f"The new Gaussian core agrees with the previous implementation at {parity['dimensions']['n']} sites, {parity['dimensions']['T']:,} fitted times and {parity['dimensions']['coefficients']:,} coefficients. The coefficient-mean relative error is {parity['checks']['coefficient_mean_relative_frobenius']:.2e}; the predictive-function covariance relative error is {parity['checks']['field_covariance_relative_frobenius']:.2e}.")
    p('A measured improvement in computation','SubClean')
    p('The development pilot exposed slow prior-centered random-walk mixing. Bounded local MAP/Laplace initialization improved proposal geometry without altering the target. A compensated horseshoe move changes global and local scales while preserving their coefficient variances exactly. Failed pilot runs are retained as evidence.')
    p('Read the comparison at its actual scale','SubClean')
    p(f"This is a compact study with {config['n_inputs']} input sites and a {config['grid_size']} by {config['grid_size']} prediction grid. It is not a new full-size comparison proving improvement over the earlier 256-site V3 forecast. The full-size result above is an algebraic compatibility check. The methods remain available for larger independently budgeted studies.")
    table(['DELIVERABLE','WHAT IT CONTAINS'],[
        ['Installable package','Observed-data containers, frozen features, readout priors, covariance families, samplers, forecasts, ensembles, calibration and scoring.'],
        ['Reproducibility','Source and numerical artifacts, explicit configurations, seeds, checkpoint state, locks, tests, wheel/source distribution and this report renderer.'],
        ['Visual evidence','Forecast/observation/error maps, trajectories, horizon diagnostics, calibration checks and three synchronized MP4 animations.'],
        ['Publication','A Git repository and ready-to-upload release assets. Remote publication status is reported separately; a configured CI file is not a completed remote CI run.']],[.25,.75])

    section(2,'The model, in layers','The response is standardized using the conditioning period. Every transformation and reservoir weight is frozen before the likelihood begins.')
    p('1. Causal nonlinear representation','SubClean')
    equation(r'$\mathbf h_t^{(l)}=(1-\boldsymbol\alpha_l)\odot\mathbf h_{t-1}^{(l)}+\boldsymbol\alpha_l\odot\tanh(W_l\mathbf h_{t-1}^{(l)}+U_l\mathbf u_t^{(l)}+\mathbf b_l)$')
    equation(r'$\mathbf z_t=\mathcal F(\mathbf y_{0:t-1};\,\text{frozen encoder and reservoir})$')
    p('Inputs can use frozen PCA, local pooling, graph diffusion or a small frozen nonlinear autoencoder. The readout includes lagged scores, multiple reservoir layers and optional bounded squared/product terms. The first reservoir sees only past observations; the upper layers see the current lower-layer state computed from that past.')
    p('2. Spatial mean and stochastic field','SubClean')
    equation(r'$\mathbf y_t=\Phi C\mathbf z_t+\mathbf w_t+\boldsymbol\epsilon_t$')
    equation(r'$\mathbf w_t\mid\sigma^2,q,\theta\sim N(0,\sigma^2 qK_\theta),\qquad\boldsymbol\epsilon_t\mid\sigma^2\sim N(0,\sigma^2 V)$')
    p('Phi is a fixed spatial dictionary evaluated at arbitrary Euclidean coordinates. Its centering and scaling use observed-site geometry. K describes spatial process dependence; V describes measurement variance. The baseline has V = I. A radial-basis width in Phi is not the covariance range in K.')
    p('3. Bayesian readout and scale','SubClean')
    equation(r'$\operatorname{vec}(C)\mid\sigma^2,\eta\sim N(0,\sigma^2\Lambda_\eta^{-1}),\qquad\sigma^2\sim\operatorname{IG}(a_0,b_0)$')
    p('The inverse-gamma convention is density proportional to v<sup>-a-1</sup> exp(-b/v). Baseline a0 = 2 and b0 = 0.05 apply on the standardized response scale. Priors remain proper, including intercept and baseline components.')
    table(['SYMBOL','ROLE'],[['n, T','Number of observed input sites and fitted time rows'],['p, d','Spatial basis and temporal feature dimensions'],['C','p by d Bayesian coefficient matrix'],['sigma², q','Nugget reference scale and spatial-to-nugget scale ratio'],['theta, eta','Covariance and readout-shrinkage hyperparameters']],[.2,.8])

    section(3,'Readout flexibility and exact inference','The reservoir is nonlinear in historical observations; conditional on the frozen observed feature matrix, coefficient inference is Gaussian.')
    table(['PRIOR FAMILY','IMPLEMENTED SPECIFICATION'],[
        ['Ridge / grouped Gaussian','One or several log-normal precision parameters. Scalar, temporal-column and grouped forms retain spectral structure.'],
        ['Regularized horseshoe','Global-local shrinkage with a finite fixed or random slab; grouping can follow temporal groups, columns or individual coefficients.'],
        ['GIGG','Group Gamma and local inverse-gamma variances, with correct generalized inverse-Gaussian group updates.'],
        ['Reference R² allocation','A Beta variance budget and Dirichlet or logistic-normal group allocation on a frozen, centered reference design. This is not a literal realized response R².'],
        ['Subspace shrinkage','A proper baseline plus residualized correction, or a proper Gaussian function penalty including its determinant.']],[.27,.73])
    equation(r'$v_j=\frac{c^2\tau^2\lambda_j^2}{c^2+\tau^2\lambda_j^2},\qquad C_{ij}\mid\cdots\sim N(0,\sigma^2v_j)$')
    p('For the Gaussian model write Y = Phi C Z-transpose + E, R = V + qK, B = Phi-transpose R-inverse Phi, G = Z-transpose Z and J = Phi-transpose R-inverse YZ. Let k = pd and N = nT.')
    equation(r'$A=\Lambda+G\otimes B,\qquad\mathbf m=A^{-1}\operatorname{vec}(J)$')
    equation(r'$Q=\operatorname{tr}(R^{-1}YY^T)-\operatorname{vec}(J)^T\mathbf m$')
    equation(r'$\sigma^2\mid Y,\eta,\theta\sim\operatorname{IG}(a_0+N/2,\ b_0+Q/2)$')
    equation(r'$\operatorname{vec}(C)\mid\sigma^2,Y,\eta,\theta\sim N(\mathbf m,\sigma^2A^{-1})$')
    p('The normalized integrated likelihood retains the prior and posterior determinants. Separable priors diagonalize small spatial and temporal matrices. The baseline never constructs the 113,004 by 113,004 coefficient precision. Near cancellation, Q is recomputed as a nonnegative residual-plus-prior quadratic.')
    p('The complete derivations, transformed densities and conditional schedules are supplied in docs/derivations/02_gaussian_priors_mcmc.md and the adjacent model chapters.','CaptionClean')

    section(4,'Spatial, temporal and robust extensions','Each extension changes a declared part of the hierarchy. Unsupported compositions fail explicitly.')
    table(['COMPONENT','MODEL AND INFERENCE CONSEQUENCE'],[
        ['Global anisotropy','A unit-determinant positive-definite metric separates directional shape from the spatial range.'],
        ['Nonstationarity','A valid Paciorek-Schervish construction permits spatially varying range, amplitude and anisotropy through a small fixed basis.'],
        ['Variance / smoothness / PC priors','Centered static log nugget variance; a finite Matérn smoothness mixture; range PC priors and an optional separate physical-scale hierarchy.'],
        ['Full-residual AR','The complete residual follows stationary AR(1). The first state and temporal determinant are included in exact whitening.'],
        ['Latent-process AR','Only the spatial process evolves as AR(1); measurement noise is iid. Spatial-mode forward filtering/backward sampling is exact under this hierarchy.'],
        ['Whole-field Student','One Gamma scale per time multiplies the complete spatial residual field.'],
        ['Local Student','Separate Gamma scales robustify site-time measurement errors; the Gaussian spatial process remains distinct.'],
        ['Predictable heteroskedasticity','Known site-time variance predictors create nonseparable observation covariances; each time-specific precision enters the readout update.']],[.27,.73])
    equation(r'$\mathbf w_t=\rho\mathbf w_{t-1}+\boldsymbol\xi_t,\quad\boldsymbol\xi_t\sim N(0,\sigma^2q(1-\rho^2)K),\quad |\rho|<1$')
    equation(r'$\omega_t\sim\operatorname{Gamma}(\nu/2,\nu/2),\quad\mathbf e_t\mid\omega_t\sim N(0,\sigma^2R/\omega_t)$')
    p('Gamma uses shape and rate. Student degrees of freedom exceed two. The whole-field marginal covariance is inflated by nu/(nu-2); the local Student model is generally a Gaussian-plus-Student convolution, not a multivariate Student field.')
    p('Ordinary conditional scale updates include coefficient normalizers: the Gaussian/whole-field/heteroskedastic shape is a0 + (N + k)/2. The latent-AR and local-Student augmented shapes are a0 + (2N + k)/2. Separate physical PC scales integrate C only and do not use the inverse-gamma collapse.')

    section(5,'The nonlinear world being forecast','Moving peaks, a traveling wave, a bending front and nonlinear phase coupling create an evolving field without using ESN features in the data generator.')
    p('Two coupled phase variables','SubClean')
    equation(r'$\theta_{t+1}=(\theta_t+0.115+0.035\sin\theta_t+0.018\sin\varphi_t)\;\mathrm{mod}\;2\pi$')
    equation(r'$\varphi_{t+1}=(\varphi_t+0.078+0.020\sin(2\theta_t-\varphi_t))\;\mathrm{mod}\;2\pi$')
    p('Both updates use the previous phases. Initial phases are independently uniform on [0, 2 pi); 500 initial updates are discarded. The retained paths differ across independent replications.')
    p('A moving spatial signal on the unit square','SubClean')
    equation(r'$g_j(s,t)=\exp\{-\|s-c_{jt}\|^2/(2d_{jt}^2)\},\quad s=(x,y)$')
    table(['COMPONENT','CENTER, WIDTH AND AMPLITUDE'],[
        ['Positive peak','c1 = (0.5 + 0.27 cos theta, 0.5 + 0.23 sin theta); d1 = 0.110 + 0.014 sin phi; a1 = 1.25 + 0.20 sin phi'],
        ['Negative peak','c2 = (0.5 + 0.28 cos(phi + 0.70), 0.5 + 0.25 sin phi); d2 = 0.135 + 0.012 cos(theta - phi); a2 = 1.05 + 0.15 cos theta']],[.23,.77])
    equation(r'$m_t(s)=a_{1t}g_1(s,t)-a_{2t}g_2(s,t)+W_t(s)+F_t(s)+I_t(s)$')
    equation(r'$W_t(s)=0.40\sin\{2\pi(1.15x-0.70y)-\theta_t\}\cos(\pi y+\varphi_t)$')
    equation(r'$F_t(s)=0.30\tanh\{9[x-0.50-0.16\sin\varphi_t-0.12\sin(2\pi y+\theta_t)]\}$')
    equation(r'$I_t(s)=0.12\sin\{3\pi(x+y)+\theta_t-2\varphi_t\}$')
    p('Spatially correlated observations','SubClean')
    equation(r'$\mathbf y_t\mid m_t\sim N(\mathbf m_t,\;0.065^2K+0.05^2I)$')
    p('K is a Matérn-3/2 correlation with length parameter 0.085. The anisotropic scenario replaces Euclidean distance by a determinant-one metric exp([[0.55, 0.25], [0.25, -0.55]]). Spatial and measurement innovations are independent across time in these two scenarios. The simulator also provides declared AR, heteroskedastic, Student, nonstationary and structural-break stress scenarios. Their API checks are separate from the two-scenario performance study.','CaptionClean')

    section(6,'A controlled forecasting study','The study was frozen after development pilots and before final-stage data were generated. Final performance never determines stopping or the within-replication model choice.')
    table(['DESIGN ITEM','LOCKED VALUE'],[
        ['Spatial domain',f"{config['grid_size']} x {config['grid_size']} regular grid; {config['n_inputs']} observed input sites; remaining sites withheld from inputs and likelihood"],
        ['Conditioning',f"Rows [0, {config['conditioning']}) for response and feature normalization"],
        ['Likelihood histories',f"{config['short_fit_times']} recent rows versus {config['long_fit_times']} rows, with the same endpoint {config['fitting_stop']}"],
        ['Validation',str(config['validation_blocks'])+'; observation CRPS at input sites selects among eligible candidates'],
        ['Final origins',str(config['final_blocks'])+'; parameters frozen, observed history updates at each origin'],
        ['Independent replications',f"{summary['replications_per_scenario']} per scenario; random initial phases plus independent named innovation streams"],
        ['Posterior / paths',f"{config['sampler']['chains']} chains, preset diagnostic-controlled draw extensions, {config['paths']} forecast paths"],
        ['Uncertainty',f"{config['bootstrap_resamples']} paired bootstrap resamples; independent simulation replication is the resampling unit"]],[.28,.72])
    p('The two data scenarios share the same nonlinear generating-mean mechanism. One has isotropic Gaussian spatial innovations and the other anisotropic innovations. All four candidate mean models are compared on identical observations and forecast dates. The short-history contrast changes the likelihood window, while preserving the same causal reservoir history.')
    p('What the larger model changes','SubClean')
    p(f"Compact reservoirs use units {config['compact_units']} and {config['compact_squares']} squared terms; expressive reservoirs use {config['expressive_units']} and {config['expressive_squares']} squared terms. Both use {config['modes']} input modes and lags {config['lags']}. The spatial mean has {config['spatial_basis_side']} x {config['spatial_basis_side']} radial centers.")
    p('Both a random-slab RHS pilot and a pilot-calibrated fixed-slab RHS sensitivity failed the strict mixing gate. Their configurations and diagnostics are retained. The final fourth candidate uses grouped Gaussian shrinkage, whose pilot passed. This development-stage decision preceded final data generation; it does not convert a failed horseshoe fit into a successful one.','CaptionClean')

    section(7,'Forecast accuracy, with failures visible','Smaller RMSE and CRPS are better. The table uses observations at the input sites; spatial interpolation performance is displayed separately in the chart.')
    table(['DATA','MODEL','PASS','RMSE','CRPS','90% COV.'],comparison_rows(summary),[.20,.26,.10,.14,.14,.16])
    p('Each independent replication contributes one score after averaging its two forecast origins. Failed diagnostic fits are excluded from validated averages and their counts are displayed. The grouped-Gaussian isotropic mean uses five replications; paired comparisons below use a common eligible set.','CaptionClean')
    p('The data generating mean is known only to evaluation. It is not passed to encoders, samplers, validation selection or forecasts. Persistence is evaluated at input sites only; no artificial spatial interpolation or probabilistic distribution is assigned to that baseline.','CaptionClean')

    section(8,'Spatial accuracy and uncertainty','The same model can improve prediction at observed input sites while worsening interpolation at withheld locations. The two domains therefore receive separate scores.')
    fig('forecast_scores.png',maxheight=425,caption='Observation RMSE and CRPS at input and withheld sites. Markers summarize eligible independent replications; failure denominators remain visible. Interval coverage is reported in the preceding table and in the lead-time diagnostics.')
    p('The larger reservoir is the clearest useful change','SubClean')
    p('At the same long likelihood history, expressive ridge improves both RMSE and CRPS in both spatial domains and both scenarios. Group-specific shrinkage has much smaller, inconsistent effects. More history alone improves input-site forecasts but slightly worsens withheld-site forecasts in these simulations.')

    section(9,'Paired effects and their uncertainty','These are CRPS differences, candidate minus comparator, at the input sites. Negative values favor the candidate. Intervals are paired bootstrap percentile intervals across independent replications.')
    table(['DATA','CONTRAST','PAIRS','DIFFERENCE [95% INTERVAL]'],paired_rows(summary),[.22,.29,.09,.40])
    p('What can be inferred','SubClean')
    p('More history, expressive features and a different prior are separate experimental contrasts. A negative point estimate alone does not establish a reliable gain when its interval crosses zero. The small replication count gives a useful controlled implementation study, with limited precision for broad empirical claims.')
    p('Conditioning on successful inference matters','SubClean')
    p('A reported paired interval uses only replications in which both compared fits passed the prespecified gate. It describes that eligible subset. Failure rates are part of the result, especially for complex models; they cannot be discarded while claiming universal superiority.')
    p('The validation-selected contrast represents an actual historical selection policy. The descriptive lowest final mean is shown for comparison only and does not retroactively alter those choices.')
    p('What this does not establish','SubClean')
    p('The experiment does not rank every implemented extension, cover all combinations, validate missing-data or irregular-time workflows, or show that the compact candidate beats the earlier much larger V3 experiment. Such claims need their own controlled studies.')

    section(10,'See the forecast evolve','The preselected matched-scenario replication is shown here. Its model was selected on validation CRPS, before final scoring.')
    fig('forecast_montage.png',maxheight=455,caption='True observations, posterior predictive observation means, and signed errors (forecast minus observation). Truth and forecast share one fixed color scale; errors use a zero-centered fixed scale. The same conventions are retained throughout the animations.')
    p(f"Illustrated model: {MODEL_LABELS.get(str(picture['selected_model']),str(picture['selected_model']))}. The illustration is one uninterrupted {len(picture['times'])}-step forecast from row {int(picture['origin'])}. It is distinct from the two shorter origins used for the primary score tables.")
    p('Each path keeps one joint coefficient/variance/covariance draw. Its own simulated observations at the input locations drive subsequent features. New target locations are sampled jointly with input locations, and duplicate coordinates share the same observation realization.')

    section(11,'Uncertainty at individual locations','Forecast intervals describe noisy observations. They are shown alongside actual observations at locations fixed in advance; the data animation separately displays the simulator generating mean.')
    fig('site_trajectories.png',maxheight=445,caption='Spatially distributed locations from the prespecified illustration. Observation variability remains even when the mean evolution is well learned. Pointwise 90% and 95% bands do not imply simultaneous trajectory coverage.')
    p('The reservoir input encoder is frozen. For the nonlinear learned-encoder option, Bayesian inference is conditional on its fitted representation; uncertainty in learned encoder weights is not sampled. Equal ensembles, validation stacking and a separate finite Bayesian reservoir-index model provide explicit alternatives for combining frozen representations.')

    section(12,'Errors and coverage over lead time','An autonomous forecast can accumulate phase or state error even when one-step residual variance is well estimated. These curves make that distinction visible.')
    fig('horizon_diagnostics.png',maxheight=390,caption='For the uninterrupted illustration: spatial observation RMSE, empirical pointwise coverage and predictive width by forecast lead. Error animations show both signed and absolute spatial errors while the running error curve advances.')
    p('Calibration is a separate historical postprocessing step','SubClean')
    p('The package supports marginal lead/site calibration, a maximum over sites at each lead, and a maximum over a complete lead/site domain. Blocks remain the sample unit: many sites and times do not turn two blocks into a large calibration sample.')
    equation(r'$r=\left\lceil(m+1)(1-\alpha)\right\rceil,\qquad q_\alpha=+\infty\ \mathrm{if}\ r>m$')
    p('The exact rank is never capped to manufacture a finite band. The API checks domain, posterior identity and chronology, and leaves raw posterior draws unchanged. The primary comparison reports raw Bayesian intervals; it does not claim an exchangeability guarantee for dependent time-series errors.')

    section(13,'Inference quality before model ranking','The experiment requires rank/folded split R-hat at most 1.01 and bulk/tail ESS at least 400, for scalar parameters and prespecified joint-recovery functionals.')
    fig('sampler_diagnostics.png',maxheight=305,caption='Diagnostic thresholds are shown explicitly. Scientific summaries include log reference variance in original units, three fixed-site readout means and the field-average readout at the final fitted feature vector. All chains retain their identities.')
    p('Adapt only during warmup','SubClean')
    p('The local MAP/Laplace calculation has a finite evaluation budget and records curvature regularization and optimizer status. It initializes chains/proposals without restricting posterior support. Retained transitions use a frozen proposal. The extra RHS move shifts log tau upward and every log local scale downward by the same amount, preserving all effective variances with a unit Jacobian.')
    p('Checkpoint continuation is substantive','SubClean')
    p('Saved state includes RNG, proposal adaptation, retained draws and the current coefficient/latent augmentation when needed. Interrupted and extended runs are compared against uninterrupted ones. Numeric JSON/NPZ round trips also preserve diagnostic column order. More retained draws are permitted; substantive transition changes cannot silently resume an old checkpoint.')

    section(14,'Independent numerical and statistical checks','Verification combines algebraic identities, posterior simulation, negative tests, public-API execution and installation checks. Passing one layer does not replace the others.')
    table(['CHECK','MEASURED EVIDENCE'],[
        ['Test suite',f"{test_count(release)} collected checks; the release manifest records status, skips, warnings and the verified environment"],
        ['Full V3 likelihood',f"Absolute normalized log-likelihood difference {parity['checks']['log_likelihood_abs']:.2e}"],
        ['Full V3 coefficient mean',f"Relative Frobenius difference {parity['checks']['coefficient_mean_relative_frobenius']:.2e}"],
        ['Full V3 covariance',f"Relative function-covariance difference {parity['checks']['field_covariance_relative_frobenius']:.2e}"],
        ['Measured full-size check',f"Peak RSS {parity['peak_rss_mib']:.1f} MiB; new likelihood evaluation {parity['timings_seconds']['new_evaluation']:.3f} seconds"],
        ['Conditional conjugate SBC',f"{sbc['replicates']} prior-predictive repetitions, {sbc['posterior_draws_per_replicate']} independent joint posterior draws each"]],[.34,.66])
    fig('sbc_calibration.png',maxheight=265,caption='Conditional conjugate simulation-based calibration. Exact 90% interval coverage for the five quantities ranges from 0.870 to 0.915; nominal coverage Monte Carlo SE is about 0.021. Every displayed Wilson 95% interval includes 0.90.')
    p('This SBC exercise uses a fixed design, covariance and proper coefficient precision. It checks joint coefficient/variance recovery and future functionals. It is not full recursive-ESN SBC and does not validate every hyperparameter or augmented sampler. Separate checks compare those MCMC transitions with independent dense formulas, distributions or quadrature.','CaptionClean')

    section(15,'Spatial approximations have two errors','A sparse solver can accurately solve its approximate model while that model remains different from the intended Matérn field. Both errors are measured.')
    fig('spatial_approximations.png',maxheight=315,caption='Vecchia approaches the exact finite-site kernel as predecessor neighbors increase. FEM accuracy depends on resolution, interpolation and domain boundaries; increasing mesh resolution alone does not remove boundary effects. Raw marginal variances are preserved, without unit-diagonal renormalization.')
    p(f"Across the 36-site reference exercise, the maximum sparse-versus-dense solve discrepancy for the same approximate model is {spatial['numerical_maximum_errors']['solve_relative']:.2e}; the log-determinant discrepancy is {spatial['numerical_maximum_errors']['logdet_absolute']:.2e}.")
    p('Implemented scope','SubClean')
    p('Vecchia keeps the fitted input ordering and appends targets. A fixed master target ordering is needed when comparing subsets. FEM uses a proper mass-lumped mesh precision and explicit interpolation. Integer operator orders 2 and 3 correspond to smoothness 1 and 2 in two dimensions, not the baseline smoothness 1.5.')
    p('The general fitting and prediction adapters still materialize bounded spatial covariance matrices. The release supplies genuine sparse precision operations, not a demonstrated end-to-end million-site MCMC system. Latent/local-Student routes requiring an inverse latent covariance reject rank-deficient coarse-mesh interpolation.')

    section(16,'An API that carries the full workflow','Observed data, model specification, posterior state, forecasting policy and evaluation truth have distinct objects and recorded identities.')
    code='''from bayes_stesn import *

data = SpatioTemporalData(y, coordinates)
spec = ModelSpec(
    features=FeatureSpec(modes=10, units=(32, 32)),
    prior=PriorSpec.grouped(),
    covariance=CovarianceSpec(),
)
prepared = prepare(data, SplitPlan(160, len(y)), spec)
fitted = fit(prepared, SamplerConfig(chains=4, warmup=1000,
                                   draws=2000, seed=17))
checks = diagnose(fitted, functional_draws=512)
if not checks["passed"]:
    raise RuntimeError("Inspect convergence before forecasting.")
prediction = simulate_predictive(
    fitted, horizon=40, target_coordinates=new_locations,
    paths=256, seed=23,
)
fitted.save("model.npz")
prediction.save("forecast.npz")'''
    story.append(Preformatted(code,styles['CodeClean']))
    p('The grouped prior above illustrates the fourth final-study family; it is not a universally optimal choice. The complete executable example and small installation smoke test are included.','CaptionClean')
    table(['OPERATION','INVARIANT'],[
        ['prepare / fit','Conditioning-only transformations; historical likelihood; no simulator truth'],
        ['advance_state','New observed history updates states while parameter posterior identity stays fixed'],
        ['refit','An explicit new parameter posterior and new data/model identity'],
        ['simulate_predictive','One parameter draw per path; simulated-observation feedback; joint spatial simulation'],
        ['fit_stacking_weights','Aligned component identities, origins and target domain inside declared validation'],
        ['fit_calibration / apply_calibration','Declared historical blocks; domain/posterior matching; no use of unavailable outcomes']],[.30,.70])

    section(17,'Reproduce, inspect, extend','The release is structured so that the evidence can be rerun and the illustrations can be regenerated without refitting.')
    table(['PATH','PURPOSE'],[
        ['src/bayes_stesn/','Numerical engines, priors, models, features, predictions, ensemble/calibration APIs'],
        ['tests/','Independent numerical, sampling, data-policy and complete-workflow checks'],
        ['docs/derivations/','Full mathematical blueprint and all extension MCMC derivations'],
        ['docs/models.md; docs/api.md','Implemented hierarchy, conventions, API and support boundaries'],
        ['configs/; examples/','Explicit model/study configurations and executable example'],
        ['benchmarks/','Protocol-locked study, Gaussian SBC, full-size parity and spatial fidelity scripts'],
        ['results/release_manifest.json','Authoritative installation/test/build evidence and environment'],
        ['report/controlled_study/','Frozen protocol, per-fit diagnostics, choices, predictions and score records'],
        ['report/render_figures.py; render_report.py','Read-only rendering from saved results; no fitting or selection']],[.42,.58])
    p('Reproducibility controls','SubClean')
    p('Pinned dependency closures, source hashes, a source distribution and a wheel are supplied. Numeric artifacts use atomic writes, safe arrays, JSON metadata, schema versions and SHA256 checksums. Random streams use explicit names rather than worker order. Installation and a complete smoke forecast are checked in a genuinely isolated environment; remote CI is configured separately.')
    p('Practical boundaries','SubClean')
    p('The initial response is scalar, input observations are complete, sites are fixed, times are equally spaced and coordinates are Euclidean. Dense augmented models have explicit size gates. Matrix-free Gaussian draws disclose finite solver tolerance and fail or use a recorded dense fallback when necessary. Representation uncertainty, covariance misspecification and structural changes can still degrade long-range forecasts.')
    p('Next scientific decision','SubClean')
    p('Use the matched comparisons and failure rates to choose a working baseline. Add anisotropy, robust noise, dynamics or nonstationarity when residual evidence and the prediction task justify them. Wider model scope is valuable; combining every extension by default is not an empirical finding.')

    section(18,'Literature and implementation map','Established constructions underpin the package. The reference-R² and ESN subspace formulations are explicit adaptations, without a claim of new theory or inherited guarantees.')
    references=[
        ('Deep spatiotemporal ESNs','McDermott and Wikle, Bayesian deep reservoir computing and deep echo-state models. Primary manuscript: https://arxiv.org/abs/1806.10728'),
        ('Regularized horseshoe','Piironen and Vehtari (2017), sparsity information and regularization. https://doi.org/10.1214/17-EJS1337SI'),
        ('GIGG shrinkage','Boss et al. (2024), group inverse-gamma gamma shrinkage. https://doi.org/10.1214/23-BA1371'),
        ('Valid nonstationary covariance','Paciorek and Schervish (2006). https://doi.org/10.1002/env.785'),
        ('Gaussian-field PC priors','Fuglstad et al. (2019). https://doi.org/10.1080/01621459.2017.1415907'),
        ('SPDE construction','Lindgren, Rue and Lindstrom (2011). https://doi.org/10.1111/j.1467-9868.2011.00777.x'),
        ('Rank-normalized diagnostics','Vehtari et al. (2021). https://doi.org/10.1214/20-BA1221'),
        ('Simulation-based calibration','Talts et al. (2018). https://arxiv.org/abs/1804.06788'),
    ]
    for title,citation in references:
        p(escape(title),'SubClean');p(escape(citation))
    p('Detailed source references and derivations appear in the model chapters. Report evidence is from this release: the controlled-study protocol and cases; V3 parity JSON; Gaussian calibration JSON/NPZ; spatial approximation JSON/NPZ; release manifest; figure and media manifests.','CaptionClean')
    p('Frozen study identity: '+protocol['protocol_sha256'],'CaptionClean')

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor(LINE));canvas.setLineWidth(.5)
        canvas.line(17*mm,16*mm,A4[0]-17*mm,16*mm)
        canvas.setFont('DejaVu',7.2);canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(17*mm,11*mm,'BAYES-STESN  |  IMPLEMENTATION & RESULTS  |  0.1.0')
        canvas.drawRightString(A4[0]-17*mm,11*mm,f'{doc.page:02d}')
        canvas.restoreState()
    document=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,
        topMargin=17*mm,bottomMargin=23*mm,title='Bayesian Spatiotemporal ESNs: Implementation and Results',
        author='bayes-stesn contributors',subject='Models, MCMC verification, controlled forecasts and reproducibility',
        pageCompression=1)
    document.build(story,onFirstPage=footer,onLaterPages=footer)


def data_url(path):
    suffix=Path(path).suffix.lower()
    mime={'.png':'image/png','.gif':'image/gif','.mp4':'video/mp4'}[suffix]
    return f'data:{mime};base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


def build_html(output,study,figures,media,values):
    summary,protocol,cases,picture,parity,sbc,spatial,release=values
    config=protocol['config']
    def html_table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def figure(filename,caption):
        return f'<figure><img loading="lazy" src="{data_url(figures/filename)}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'
    def video(filename,title,caption):
        return f'<section class="video"><h3>{escape(title)}</h3><video controls playsinline muted preload="metadata" src="{data_url(media/filename)}"></video><p class="caption">{escape(caption)}</p></section>'
    content=[]
    content.append('<header><div class="eyebrow">BAYES-STESN / RESEARCH RELEASE 0.1.0</div><h1>Learning<br>evolving fields.</h1><p class="lead">Bayesian spatiotemporal echo state networks, from hierarchy to verified forecasts.</p><div class="badges"><span>23 extension IDs</span><span>'+str(test_count(release))+' checks</span><span>'+str(summary['replications_per_scenario'])+' replications per scenario</span></div></header>')
    content.append('<nav><a href="#findings">Findings</a><a href="#forecast">Watch the forecast</a><a href="#accuracy">Comparisons</a><a href="#uncertainty">Uncertainty</a><a href="#verification">Verification</a><a href="#reproduce">Reproduce</a></nav>')
    content.append('<main><section id="findings"><div class="eyebrow">01 / WHAT THE EVIDENCE SAYS</div><h2>More capability.<br>Measurable checks.</h2>'+''.join('<p>'+escape(x)+'</p>' for x in findings(summary))+'</section>')
    content.append('<p>The larger reservoir produced the clearest gain. More history alone slightly worsened withheld-site accuracy, and group-specific shrinkage gave no consistent advantage over expressive ridge. Both horseshoe development pilots failed the strict mixing gate; their failure records are retained.</p>')
    content.append('<section id="forecast"><div class="eyebrow">02 / AN AUTONOMOUS FORECAST</div><h2>Truth, forecast, error.</h2><p>The illustration is the prespecified matched-scenario replication 0. Its model was chosen on validation CRPS. The uninterrupted 80-step movie is distinct from the two 40-step origins used in the primary score tables.</p>')
    content.append(video('forecast_comparison.mp4','Forecast beside true observations','One parameter draw per path; simulated observations feed future reservoir inputs. Truth/forecast scales and zero-centered error scales stay fixed throughout.'))
    content.append(video('data_evolution.mp4','How the nonlinear data evolve','Observed fields are labeled separately from any generating-mean reference. The simulator truth is used only for evaluation and presentation.'))
    content.append(video('error_evolution.mp4','Where error accumulates','Signed error is predictive mean minus observation. Absolute error and the advancing RMSE curve show spatial and temporal behavior.'))
    content.append('</section><section id="accuracy"><div class="eyebrow">03 / CONTROLLED COMPARISONS</div><h2>Separate the changes.</h2><p>The same sites, endpoints, conditioning period and random data streams isolate more likelihood history, more reservoir features and a different shrinkage prior. Four-chain diagnostics screen eligible results; failed fits remain visible.</p>')
    content.append(html_table(['Data','Model','Eligible','RMSE','CRPS','90% coverage'],comparison_rows(summary)))
    content.append(figure('forecast_scores.png','Input-site and withheld-site forecast metrics, with independent-replication uncertainty and explicit failure counts.'))
    content.append(html_table(['Data','Contrast','Paired replications','CRPS difference [95% interval]'],paired_rows(summary)))
    content.append('<p class="caption">Negative differences favor the candidate. Paired intervals condition on both fits passing diagnostics. A descriptive final-score winner is not a replacement for the historical validation-selection policy.</p></section>')
    content.append('<section id="uncertainty"><div class="eyebrow">04 / WHAT THE BANDS MEAN</div><h2>Uncertainty has a target.</h2>')
    content.append(figure('site_trajectories.png','Observation predictions and intervals at spatially distributed sites. These are pointwise bands, not simultaneous path bands.'))
    content.append(figure('horizon_diagnostics.png','Observation errors, empirical coverage and interval width along the uninterrupted illustrative forecast.'))
    content.append('<p>Calibration uses historical blocks with explicit domain and chronology checks. The exact conformal rank returns an infinite band when the reserve is too small. It never treats many sites and leads as independent blocks, and it does not change the raw posterior paths.</p></section>')
    content.append('<section id="verification"><div class="eyebrow">05 / INDEPENDENT EVIDENCE</div><h2>Check the mathematics.<br>Check the workflow.</h2>')
    content.append(figure('sampler_diagnostics.png','Rank-normalized/folded R-hat and bulk/tail ESS for hyperparameters and prespecified readout functionals.'))
    content.append(f'<p>The full-size V3 check uses {parity["dimensions"]["coefficients"]:,} coefficients. Its coefficient-mean relative discrepancy is {parity["checks"]["coefficient_mean_relative_frobenius"]:.2e}; the field-function covariance discrepancy is {parity["checks"]["field_covariance_relative_frobenius"]:.2e}. This verifies algebraic compatibility, not a new full-size forecast comparison.</p>')
    content.append(figure('sbc_calibration.png','200 conditional conjugate Gaussian SBC replications. This does not claim full recursive-ESN calibration.'))
    content.append(figure('spatial_approximations.png','Approximation fidelity is measured separately from sparse linear-algebra accuracy.'))
    content.append('</section><section id="reproduce"><div class="eyebrow">06 / A COMPLETE RESEARCH PROJECT</div><h2>Inspect every step.</h2><p>The repository includes the full mathematical derivations, installable package, explicit model configurations, numerical tests, isolated-installation checks, immutable study protocol, saved numeric evidence and rendering source. This page is self-contained and requires no network requests to play the media.</p><pre>python -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-dev.lock\npython -m pip install --no-deps -e .\npython -m pytest tests\nbstesn demo --smoke --output results/smoke</pre><p>The compact study does not establish superiority over the earlier much larger V3 forecast, optimality across data sets, or scalability to millions of sites. Sparse primitives are implemented; the initial general fitting adapters retain explicit dense limits. The detailed PDF and model documentation state all supported and experimental combinations.</p><p class="hash">Frozen protocol SHA256<br>'+protocol['protocol_sha256']+'</p></section></main><footer>02 November 2026 · bayes-stesn contributors · MIT-licensed project source · Generated from saved results only.</footer>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f8f8;color:#263746;font:17px/1.7 system-ui,-apple-system,"DejaVu Sans",sans-serif}header{background:#14283b;color:#fff;padding:80px max(6vw,calc((100vw - 1100px)/2)) 65px}h1{font-size:clamp(58px,8vw,112px);line-height:.99;letter-spacing:-.055em;margin:22px 0 28px}h2{font-size:clamp(34px,5vw,58px);line-height:1.12;letter-spacing:-.04em;color:#14283b;margin:15px 0 28px}h3{font-size:26px;letter-spacing:-.02em;line-height:1.3}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.17em;color:#007f88}header .eyebrow{color:#8bd7d8}.lead{font-size:23px;max-width:700px;color:#d8e9ec}.badges{display:flex;gap:12px;flex-wrap:wrap;margin-top:30px}.badges span{border:1px solid #52727c;border-radius:30px;padding:7px 15px;font-size:13px}nav{position:sticky;top:0;z-index:2;display:flex;gap:24px;padding:16px max(5vw,calc((100vw - 1100px)/2));overflow-x:auto;background:#fff;border-bottom:1px solid #d8e2e5}nav a{font-size:13px;font-weight:700;white-space:nowrap;color:#007f88;text-decoration:none}main{max-width:1100px;margin:auto;padding:10px 32px 50px}main>section{padding:65px 0;border-bottom:1px solid #d8e2e5;scroll-margin-top:65px}p{max-width:950px}figure{margin:30px 0 35px;padding:20px;background:#fff;border:1px solid #e2e9eb;border-radius:14px}figure img{display:block;width:100%;height:auto}figcaption,.caption{font-size:13px;color:#596d7a;line-height:1.6;margin:15px 5px 0}video{width:100%;border-radius:12px;background:#14283b}.video{padding:15px 0 35px}.scroll{overflow:auto;margin:25px 0}table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}th{text-align:left;background:#14283b;color:#fff;font-size:11px;letter-spacing:.04em;padding:14px 12px}td{padding:13px 12px;border-bottom:1px solid #e5ebed}tr:nth-child(even){background:#eef4f5}pre{background:#14283b;color:#dcecef;border-radius:12px;padding:28px;overflow:auto;font-size:13px;line-height:1.8}.hash{font-family:monospace;font-size:12px;word-break:break-all;color:#596d7a}footer{background:#14283b;color:#cadbdf;text-align:center;padding:35px;font-size:12px}@media(max-width:600px){main{padding:5px 18px}header{padding:60px 24px}.lead{font-size:19px}nav{padding:14px 20px;gap:20px}figure{padding:8px}main>section{padding:45px 0}}'''
    output.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bayesian Spatiotemporal ESNs · Implementation & Results</title><style>'+css+'</style></head><body>'+''.join(content)+'</body></html>',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'report'/'controlled_study')
    parser.add_argument('--output',type=Path,default=ROOT/'report')
    parser.add_argument('--assets',type=Path,default=ROOT/'report',
                        help='Directory containing figures/ and media/ generated by render_figures.py')
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    figures=args.assets/'figures';media=args.assets/'media'
    values=evidence(args.study)
    pdf=args.output/'bayes_stesn_implementation_report.pdf'
    html=args.output/'bayes_stesn_results.html'
    build_pdf(pdf,args.study,figures,values)
    build_html(html,args.study,figures,media,values)
    manifest={'protocol_sha256':values[1]['protocol_sha256'],
              'renderer_sha256':sha(Path(__file__)),
              'outputs':{str(path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path.resolve()):{'sha256':sha(path),'bytes':path.stat().st_size} for path in (pdf,html)},
              'policy':'Read-only rendering of saved results; no refitting, selection or recalibration'}
    (args.output/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
