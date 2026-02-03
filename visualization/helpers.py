import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

def load_result_table(epoch=320,
                      ema = "_ema",
                loop_params = {
                    "AU": ["softmax", "ssn", "diffusion" ],
                    "EU": [ "swag_diag", "swag", "dropout", "ensemble"],
                    "network": [ "unet-s", "unet-m"],
                },
                is_ood_aug = True,
                formatter = "{EU}_{AU}_{network}_lidc_2d_small",
                save_path = "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/test_results/",
                aggregation_type="patch_level"):
    assert aggregation_type in ["patch_level", "threshold"], "Invalid aggregation type"
    table = pd.DataFrame()

    for values in product(*loop_params.values()):
        add_dict = dict(zip(loop_params.keys(),values))
        version = formatter.format(**add_dict).replace("ensemble","ens12[1,2,3,4,5]")
        if not (Path(f"{save_path}")/version).exists():
            print("Skipping missing version:", version)
            continue
        add_dict["version"] = version
        p = f"{save_path}{version}/e{epoch}{ema}/val/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["val_dice"] = loaded["mean"]["metrics"]["dice"]
        add_dict["val_ged"] = loaded["mean"]["metrics"]["ged"]
        add_dict["val_ged_bma"] = loaded["mean"]["metrics"]["ged_bma"]
        
        p = f"{save_path}{version}/e{epoch}{ema}/id/metrics.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["id_dice"] = loaded["mean"]["metrics"]["dice"]
        add_dict["id_ged"] = loaded["mean"]["metrics"]["ged"]
        add_dict["id_ged_bma"] = loaded["mean"]["metrics"]["ged_bma"]
        
        p = f"{save_path}{version}/e{epoch}{ema}/id/ambiguity_modeling.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["EU_ncc_id"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ncc"]
        add_dict["AU_ncc_id"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ncc"]
        add_dict["TU_ncc_id"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ncc"]
        p = f"{save_path}{version}/e{epoch}{ema}/id/calibration.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        add_dict["EU_ace_id"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ace"]
        add_dict["AU_ace_id"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ace"]
        add_dict["TU_ace_id"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ace"]
        p = f"{save_path}{version}/e{epoch}{ema}/ood_detection.json"
        with open(p, "r") as f:
            loaded = json.load(f)
        if is_ood_aug:
            for k in loaded.keys():
                k2 = k.replace("id&","")
                add_dict[f"EU_auc_{k2}"] = loaded[k]["mean"]["epistemic_uncertainty"][aggregation_type]["metrics"]["auroc"]
                add_dict[f"AU_auc_{k2}"] = loaded[k]["mean"]["aleatoric_uncertainty"][aggregation_type]["metrics"]["auroc"]
                add_dict[f"TU_auc_{k2}"] = loaded[k]["mean"]["predictive_uncertainty"][aggregation_type]["metrics"]["auroc"]

            for p in Path(f"{save_path}{version}/e{epoch}{ema}").glob("ood*/ambiguity_modeling.json"):
                k2 = p.parts[-2]
                with open(p, "r") as f:
                    loaded = json.load(f)
                add_dict[f"EU_ncc_{k2}"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ncc"]
                add_dict[f"AU_ncc_{k2}"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ncc"]
                add_dict[f"TU_ncc_{k2}"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ncc"]
            for p in Path(f"{save_path}{version}/e{epoch}{ema}").glob("ood*/calibration.json"):
                k2 = p.parts[-2]
                with open(p, "r") as f:
                    loaded = json.load(f)
                add_dict[f"EU_ace_{k2}"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ace"]
                add_dict[f"AU_ace_{k2}"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ace"]
                add_dict[f"TU_ace_{k2}"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ace"]
        else:
            add_dict["EU_auc"] = loaded["mean"]["epistemic_uncertainty"][aggregation_type]["metrics"]["auroc"]
            add_dict["AU_auc"] = loaded["mean"]["aleatoric_uncertainty"][aggregation_type]["metrics"]["auroc"]
            add_dict["TU_auc"] = loaded["mean"]["predictive_uncertainty"][aggregation_type]["metrics"]["auroc"]
            p = f"{save_path}{version}/e{epoch}{ema}/ood/ambiguity_modeling.json"
            with open(p, "r") as f:
                loaded = json.load(f)
            add_dict["EU_ncc_ood"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ncc"]
            add_dict["AU_ncc_ood"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ncc"]
            add_dict["TU_ncc_ood"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ncc"]
            p = f"{save_path}{version}/e{epoch}{ema}/ood/calibration.json"
            with open(p, "r") as f:
                loaded = json.load(f)
            add_dict["EU_ace_ood"] = loaded["mean"]["epistemic_uncertainty"]["metrics"]["ace"]
            add_dict["AU_ace_ood"] = loaded["mean"]["aleatoric_uncertainty"]["metrics"]["ace"]
            add_dict["TU_ace_ood"] = loaded["mean"]["predictive_uncertainty"]["metrics"]["ace"]
        table = pd.concat([table, pd.DataFrame([add_dict])], ignore_index=True)
    valid_ood_keys = []
    for k in table.columns:
        if k.startswith("EU_auc_ood"):
            valid_ood_keys.append(k.replace("EU_auc_",""))
    for k2 in valid_ood_keys:
        table[f"(AU-EU)_ncc_{k2}"] = (table[f"AU_ncc_{k2}"] - table[f"EU_ncc_{k2}"])/table[f"AU_ncc_{k2}"]
        table[f"(EU-AU)_auc_{k2}"] = (table[f"EU_auc_{k2}"] - table[f"AU_auc_{k2}"])/table[f"EU_auc_{k2}"]
        table[f"(min(AU,EU)-TU)_ace_{k2}"] = table[[f"AU_ace_{k2}", f"EU_ace_{k2}"]].min(axis=1) - table[f"TU_ace_{k2}"]
    #same as above but for non-ood keys
    table[f"(AU-EU)_ncc_id"] = (table["AU_ncc_id"] - table["EU_ncc_id"])/table["AU_ncc_id"]
    table[f"(min(AU,EU)-TU)_ace_id"] = (table[["AU_ace_id", "EU_ace_id"]].min(axis=1) - table["TU_ace_id"])/table["TU_ace_id"]
    return table
