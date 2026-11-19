# Evaluation results

Runs: 30 · systems: mas, mas-cheap, single

## Aggregate

| Metric | mas | mas-cheap | single |
|---|---|---|---|
| Top-1 accuracy | 1 | 0.9 | 1 |
| Top-3 accuracy | 1 | 1 | 1 |
| Outcome correctness | 1 | 0.8 | 0.8 |
| Confident-wrong rate ↓ | 0 | 0 | 0 |
| Citation validity | 1 | 1 | 1 |
| Claim coverage (cited) | 0.8863 | 0.8836 | 1 |
| Key-evidence recall | 1 | 1 | 1 |
| Brier score ↓ | 0.1644 | 0.2367 | 0.0562 |
| Rounds (mean) | 1.8 | 1.9 | 1 |
| Tool calls (mean) | 34.7 | 35.1 | 39.3 |
| LLM calls (mean) | 22.6 | 24.1 | 3.5 |
| Input tokens (mean) | 5.914e+04 | 5.975e+04 | 2.518e+04 |
| Output tokens (mean) | 1.129e+04 | 1.524e+04 | 3404 |
| Cost per run ($) | 0.1087 | 0.057 | 0.0362 |
| Latency (s, mean) | 147 | 158.6 | 51.34 |
| Cache hit rate (cached/input) | 0.1215 | 0.176 | 0.2376 |
| Model fallbacks (mean) | 0 | 0 | 0 |
| Cost per correct diagnosis ($) | 0.1087 | 0.0633 | 0.0362 |
| Crashed runs | 0 | 0 | 0 |

## Tokens by agent (mean per run)

| Agent | mas | mas-cheap | single |
|---|---|---|---|
| critic | 7697 | 7580 | 0 |
| dependency | 13141 | 15637 | 0 |
| observability | 29194 | 30797 | 0 |
| orchestrator | 3802 | 3793 | 0 |
| reasoning | 11465 | 12249 | 0 |
| reporter | 5136 | 4930 | 0 |
| single | 0 | 0 | 28583 |

## Top-1 accuracy by scenario trait

| Trait | mas | mas-cheap | single |
|---|---|---|---|
| conflicting | 1.00 (n=3) | 0.67 (n=3) | 1.00 (n=3) |
| control | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| inconclusive | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| misleading | 1.00 (n=5) | 1.00 (n=5) | 1.00 (n=5) |
| noisy | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| tool_failure | 1.00 (n=3) | 1.00 (n=3) | 1.00 (n=3) |

## Per-scenario results

| Scenario | System | Outcome | Predicted | Truth | Top-1 | Outcome ok | KE recall | Tools | Tokens | $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S01 | mas | ROOT_CAUSE_IDENTIFIED | checkout/bad_deploy | checkout/bad_deploy | ✅ | ✅ | 1.00 | 33 | 76441 | 0.1138 | 154.2 |
| S01 | mas-cheap | ROOT_CAUSE_IDENTIFIED | checkout/bad_deploy | checkout/bad_deploy | ✅ | ✅ | 1.00 | 32 | 71211 | 0.0536 | 143.3 |
| S01 | single | ROOT_CAUSE_IDENTIFIED | checkout/bad_deploy | checkout/bad_deploy | ✅ | ✅ | 1.00 | 40 | 27897 | 0.0406 | 59.6 |
| S02 | mas | PROBABLE | search/config_change | search/config_change | ✅ | ✅ | 1.00 | 33 | 70219 | 0.1081 | 159.9 |
| S02 | mas-cheap | PROBABLE | search/config_change | search/config_change | ✅ | ✅ | 1.00 | 40 | 85842 | 0.0619 | 156.0 |
| S02 | single | INCONCLUSIVE | search/config_change | search/config_change | ✅ | ❌ | 1.00 | 40 | 23529 | 0.0315 | 42.6 |
| S03 | mas | PROBABLE | payments-db/resource_exhaustion | payments-db/resource_exhaustion | ✅ | ✅ | 1.00 | 40 | 76197 | 0.1241 | 171.6 |
| S03 | mas-cheap | INCONCLUSIVE | payments-db/resource_exhaustion | payments-db/resource_exhaustion | ✅ | ❌ | 1.00 | 35 | 83091 | 0.0616 | 176.9 |
| S03 | single | INCONCLUSIVE | payments-db/resource_exhaustion | payments-db/resource_exhaustion | ✅ | ❌ | 1.00 | 40 | 36018 | 0.0420 | 52.7 |
| S04 | mas | ROOT_CAUSE_IDENTIFIED | auth/resource_exhaustion | auth/resource_exhaustion | ✅ | ✅ | 1.00 | 40 | 83944 | 0.1273 | 156.1 |
| S04 | mas-cheap | PROBABLE | auth/resource_exhaustion | auth/resource_exhaustion | ✅ | ✅ | 1.00 | 40 | 82891 | 0.0651 | 177.9 |
| S04 | single | ROOT_CAUSE_IDENTIFIED | auth/resource_exhaustion | auth/resource_exhaustion | ✅ | ✅ | 1.00 | 40 | 30177 | 0.0341 | 44.6 |
| S05 | mas | PROBABLE | api-gateway/traffic_surge | inventory/traffic_surge | ✅ | ✅ | 1.00 | 17 | 41753 | 0.0703 | 94.0 |
| S05 | mas-cheap | PROBABLE | inventory/traffic_surge | inventory/traffic_surge | ✅ | ✅ | 1.00 | 39 | 88462 | 0.0681 | 181.1 |
| S05 | single | ROOT_CAUSE_IDENTIFIED | inventory/traffic_surge | inventory/traffic_surge | ✅ | ✅ | 1.00 | 40 | 25062 | 0.0375 | 55.4 |
| S06 | mas | PROBABLE | cache/config_change | cache/config_change | ✅ | ✅ | 1.00 | 40 | 75406 | 0.1199 | 154.2 |
| S06 | mas-cheap | ROOT_CAUSE_IDENTIFIED | cache/config_change | cache/config_change | ✅ | ✅ | 1.00 | 40 | 71719 | 0.0597 | 143.8 |
| S06 | single | ROOT_CAUSE_IDENTIFIED | cache/config_change | cache/config_change | ✅ | ✅ | 1.00 | 40 | 23357 | 0.0333 | 46.2 |
| S07 | mas | PROBABLE | payments/infrastructure | payments/infrastructure | ✅ | ✅ | 1.00 | 38 | 72197 | 0.1075 | 176.5 |
| S07 | mas-cheap | INCONCLUSIVE | payments/unknown | payments/infrastructure | ❌ | ❌ | 1.00 | 28 | 63113 | 0.0483 | 163.0 |
| S07 | single | ROOT_CAUSE_IDENTIFIED | payments/infrastructure | payments/infrastructure | ✅ | ✅ | 1.00 | 40 | 26156 | 0.0363 | 51.7 |
| S08 | mas | ROOT_CAUSE_IDENTIFIED | queue/resource_exhaustion | queue/resource_exhaustion | ✅ | ✅ | 1.00 | 26 | 45302 | 0.0701 | 84.7 |
| S08 | mas-cheap | ROOT_CAUSE_IDENTIFIED | queue/resource_exhaustion | queue/resource_exhaustion | ✅ | ✅ | 1.00 | 24 | 41696 | 0.0300 | 84.9 |
| S08 | single | ROOT_CAUSE_IDENTIFIED | queue/resource_exhaustion | queue/resource_exhaustion | ✅ | ✅ | 1.00 | 33 | 35296 | 0.0307 | 51.1 |
| S09 | mas | INCONCLUSIVE | user-svc/unknown | user-svc/dependency_failure | ✅ | ✅ | 1.00 | 40 | 75468 | 0.1151 | 151.8 |
| S09 | mas-cheap | INCONCLUSIVE | user-svc/resource_exhaustion | user-svc/dependency_failure | ✅ | ✅ | 1.00 | 37 | 76828 | 0.0596 | 169.6 |
| S09 | single | INCONCLUSIVE | user-svc/unknown | user-svc/dependency_failure | ✅ | ✅ | 1.00 | 40 | 32111 | 0.0381 | 55.3 |
| S10 | mas | PROBABLE | payments/config_change | payments/config_change | ✅ | ✅ | 1.00 | 40 | 87423 | 0.1311 | 166.9 |
| S10 | mas-cheap | PROBABLE | payments/config_change | payments/config_change | ✅ | ✅ | 1.00 | 36 | 85010 | 0.0617 | 189.8 |
| S10 | single | ROOT_CAUSE_IDENTIFIED | payments/config_change | payments/config_change | ✅ | ✅ | 1.00 | 40 | 26230 | 0.0376 | 54.1 |
