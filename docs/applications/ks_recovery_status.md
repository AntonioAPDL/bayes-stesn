# KS application: completed computation and incomplete original artifact recovery

## Status

The original campaign reached its terminal schedule on 2026-11-05 at 05:51:54 UTC. The recorded final ledger contained 900 terminal jobs: 870 complete, 29 failed, and one inference-unqualified. The completed schedule included both main tracks, the prespecified extra model seeds, and contiguous-site robustness. Scientific failures remain failures; terminal execution does not mean successful forecasting.

Before the final report render and full result-backup upload, the original execution workspace became unavailable. Read-only checks of its known location and possible exposed recovery locations did not recover it. The cause and recoverability of the original execution host are not established. No restoration of the old directory, model reset, result substitution, or repeated full campaign has been performed.

**The original per-origin results, most fitted models, full chain/optimizer checkpoints, final extension-audit file, original example predictions, and observer/log archive are currently unavailable. The complete original result package must not be described as delivered.** The command-line tools for producing that package remain in this branch.

## Evidence that is available

| Evidence | Recovery status |
| --- | --- |
| Published implementation at `7802ddddc2c0cf27cc54ce00490e3d0affd8f681` | Recovered from the existing application branch; tree `61777a154c276cb5a51312d6ab044a963399eb06`. |
| Accepted core model | All 43 authoritative core source files preserved; all 51 frozen numerical source/configuration bindings checked. |
| Main A/B independent audit | Original 254,889-byte published JSON recovered exactly; SHA-256 `fe4448818eda7e7f3ead715d989e5900db5464d47ddd446ce357c161e792f4c0`. Contains 17 model rows, the six predeclared lead summaries, and full-schedule reliability counts. |
| Complete frozen specification | Original 33,020-byte private backup recovered exactly; SHA-256 `90e9a5e82e077eae15c19ae887496f1ec4b4fc706925d961ae539852efd699e8`. |
| A / trajectory 0 / main-seed Bayesian fitted model | Original 34,430,031-byte private backup recovered exactly; SHA-256 `cfe7b6598deaf76beb7198232ac8a98c17eb22861956cf811808b78ea0d38049`. |
| KS input data and delivered readers | Retrieved again through the authorized connection; all four original byte counts and SHA-256 values match. Only the KS v1 and source v2 payloads were restored. |
| Final HTML-report and presentation-test source | Reconstructed from retained exact patches; hashes match the pre-outage files byte-for-byte. |
| Final PDF-renderer source | Retained final changes restored, with two evidence-binding helpers reconstructed. The resulting source is explicitly not byte-identical to the original final renderer. |
| Contiguous/seed results and spatial diagnosis | Selected exact numbers, recorded audit outcomes, and original file hashes survive in the conversation. New JSON summaries explicitly identify this transcript-derived evidence; they are not replacements for the missing original audits. |

The earlier 347,338,663-byte validation-checkpoint upload returned an uncertain timeout. Subsequent read-only lookup did not find a completed file. It has not been retried or represented as a successful backup.

## Scientific conclusions supported by the retained evidence

In the strict interleaved track, the Bayesian model had RMSE 1.429798 and CRPS 0.844727. The separable GP and basis GRU both had lower RMSE and CRPS, while the linear/POD VAR had RMSE 0.990674. The Bayesian model's aggregate interval coverage conceals marked early-horizon undercoverage. Its paired RMSE difference from the conventional ridge ESN has a cluster interval spanning zero.

In the all-observed track, the small circular CNN had RMSE 0.667456; the Bayesian model had RMSE 1.225467. The Bayesian model again had worse RMSE and CRPS than the GP and GRU. Its small mean RMSE advantage over persistence and the ridge ESN is not resolved by the retained trajectory-level intervals. POD-NVAR achieved small typical errors but rare extreme forecasts substantially worsened RMSE; these values were not clipped.

The recorded contiguous robustness result is a severe failure of the transferred Bayesian configuration: RMSE 2106.688962, CRPS 905.339301, and 400 finite-divergent origins out of 400. All 20 Bayesian fits passed the recorded convergence gate. Qualification of posterior simulation therefore did not establish predictive adequacy. The retained fixed-case diagnosis supports amplification in the unseen-site spatial representation; it is not a complete causal proof or a corrected model. Contiguous POD-NVAR had 260 nonfinite origins, 37 additional finite divergences, and no complete trajectory, so its aggregate accuracy is undefined.

These are results of a custom, literature-grounded KS experiment, conditional on its selected configurations. They are not a claim about universal model superiority or a reproduction of a published leaderboard.

## Recovery deliverables and reproduction

A separate recovery-report generator uses the published main audit directly and labels the additional transcript-derived evidence. It must not manufacture a runner summary or bypass the original final-report evidence checks. Any example reforecast from the retained Bayesian fit is a new, identified computation with the frozen case, configuration, history, path count and random stream; it is not described as recovery of the missing original prediction bytes.

The main reproduction workflow is documented in [ks_reproduction.md](ks_reproduction.md). The [backup/restore instructions](ks_backup_and_restore.md) apply only to a package actually created and verified. At present, no complete final multipart backup exists.

To finish the original artifact handoff, restore access to the original workspace or a verified complete copy, then run the evidence collector, original report generator, PDF checks, and packaging commands against those real results. If that workspace cannot be recovered, recreating the unavailable fits and forecasts is a new, explicitly budgeted reproduction of the frozen study; hashes, dates and runtime must be recorded as new output. It must not silently replace the original result provenance or reset the original campaign's six recorded worker hours.

Recovery receipts in `docs/applications/evidence/ks_recovered_*.json` distinguish exact source recovery, reconstructed source, and transcript-derived scientific facts. Private storage identifiers, account details, catalogs and bulk data remain outside the repository.
