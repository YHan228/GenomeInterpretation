import pandas as pd, os
for name, fn in [("curated","sporulation/analysis_out/family_enrichment_curated.csv"),
                 ("data_driven","sporulation/analysis_out/family_enrichment_data_driven.csv")]:
    f = pd.read_csv(fn)
    print(name, {
        "total": len(f),
        "finite_fdr": int(f["fdr_bh"].notna().sum()),
        "zeros_fdr": int((f["fdr_bh"]==0).sum()),
        "sig_fdr_<=0.05": int((f["fdr_bh"]<=0.05).sum()),
        "nan_p_cmh": int(f["p_cmh_one_sided"].isna().sum()) if "p_cmh_one_sided" in f else "n/a"
    })