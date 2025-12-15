import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_status_report(df: pd.DataFrame):
    status_cols = ['observation_status', 'OBS_STATUS2', 'OBS_STATUS3']
    existing = [c for c in status_cols if c in df.columns]
    mask_d = df[existing].eq('D').any(axis=1) if existing else pd.Series(False, index=df.index)

    per_code_total = df.groupby('pharmaceutical_code').size()
    per_code_d = df[mask_d].groupby('pharmaceutical_code').size()

    report = (
        pd.DataFrame({
            'rows_with_D': per_code_d,
            'rows_total': per_code_total,
        })
        .fillna(0)
        .assign(share_D=lambda x: x['rows_with_D'] / x['rows_total'].replace(0, pd.NA))
        .fillna({'share_D': 0})
        .sort_values('share_D', ascending=True)
    )
    return report, mask_d.sum()

def plot_share_d(report: pd.DataFrame):
    top5 = report.reset_index().rename(columns={'index': 'pharmaceutical_code'}).sort_values('share_D').head(5)
    shares = top5['share_D']
    if shares.max() > shares.min():
        norm = (shares - shares.min()) / (shares.max() - shares.min())
    else:
        norm = np.zeros(len(shares))
    # Grün, niedrige Quote = kräftig, hohe Quote = transparenter
    base_color = np.array([0/255, 158/255, 115/255, 1.0])
    colors = []
    for v in norm:
        alpha = 1.0 - v  # höherer Anteil -> mehr Transparenz
        c = base_color.copy(); c[3] = alpha; colors.append(tuple(c))

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top5,
        y='pharmaceutical_code', x='share_D',
        palette=colors
    )
    plt.xlabel('Anteil D-Flags (share_D)')
    plt.ylabel('ATC-Code')
    plt.title('D-Flags je ATC-Code (Top 5, niedriger Anteil = kräftiger grün)')
    plt.tight_layout()
    plt.show()

report, total_d = compute_status_report(df)
print('\nRows mit D (observation_status / OBS_STATUS2 / OBS_STATUS3) pro ATC-Code (Top 5):')
print(report.head(5).to_string())
print(f'\nTotal rows with D: {total_d:,}')

plot_share_d(report)


