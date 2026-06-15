# HCP Segmentation

This project starts with synthetic healthcare-provider data for segmentation experiments.

The generator creates four CSV files that can be loaded directly with pandas:

```python
import pandas as pd

claims_data = pd.read_csv("data/claims_data.csv")
ehr_data = pd.read_csv("data/ehr_data.csv")
provider_data = pd.read_csv("data/provider_data.csv")
call_activity = pd.read_csv("data/call_activity.csv")
```

It also creates `data/hcp_segmentation_synthetic_data.xlsx`, with one sheet per dataset and a codebook sheet for easier manual review in Excel.

## Generate Data

From `backend` with the virtual environment activated:

```bash
python hcp_segmentation/scripts/generate_synthetic_data.py
```

The data is fully synthetic and should not be treated as clinical or claims guidance.

