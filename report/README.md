# Reports and saved forecasting evidence

The full release download contains the polished PDF, a self-contained HTML
report with embedded videos, three MP4 animations, a compact GIF preview and
the numerical evidence used to render them. The report renderers do not fit,
select, recalibrate or rescore models.

## Render the saved evidence

Install `requirements-report.lock`. The media renderer also needs FFmpeg and
FFprobe available on `PATH`.

```bash
python report/render_figures.py --study report/controlled_study --output report
python report/render_report.py --study report/controlled_study --assets report --output report
```

The output directory contains:

- `bayes_stesn_implementation_report.pdf`: hierarchy, inference, measured results,
  verification, practical scope and reproducibility.
- `bayes_stesn_results.html`: an offline report with embedded plots and videos.
- `media/data_evolution.mp4`: observed fields and the generating-mean reference.
- `media/forecast_comparison.mp4`: observations, forecast mean and signed error,
  with advancing error and coverage curves.
- `media/error_evolution.mp4`: signed/absolute spatial errors and lead-time errors.
- `figures/`: high-resolution figures and their numerical source manifest.

The illustrated run is fixed in advance: matched scenario, replication 0, with
the model chosen using validation CRPS. It is an uninterrupted 80-step forecast
with 256 posterior paths. The main comparison uses two 40-step origins and
128 paths per origin. These are deliberately labeled as different evaluations.

## Scientific records

`controlled_study/protocol.json` fixes data, feature dimensions, candidate
models, validation/final blocks, diagnostics, source hashes and stopping rules.
`summary.json` aggregates completed cases. Each final case retains its scores,
diagnostics, validation selection and saved forecast summaries. The two
preselected illustration cases also retain full paths and fitted-model files.

The development directories retain failed prior-centered and horseshoe pilot
results. They are not included in final-performance averages. Atomic temporary
files, environments and render QA contact sheets are omitted from releases.

Large generated arrays and downloadable reports live in the release archive;
the Git repository carries the source, curated figures and compact numerical
records. Reproduction of the full-size historical V3 parity additionally needs
the original V3 artifacts; the bundled compact parity fixture is self-contained.

