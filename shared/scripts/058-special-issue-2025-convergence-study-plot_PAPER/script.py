import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_data(filepath):
    """Load dataset from text file while skipping commented rows."""
    with open(filepath, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#') and line.strip()]

    from io import StringIO
    data_str = ''.join(lines)

    try:
        df = pd.read_csv(StringIO(data_str), delim_whitespace=True)
    except Exception:
        df = pd.read_csv(StringIO(data_str), sep='\t')

    # --- Convert numeric columns automatically ---
    for col in df.columns:
        # Convert to numeric, coercing errors (like '-' -> NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Optionally drop rows where nu_cells is NaN ---
    if 'nu_cells' in df.columns:
        df = df.dropna(subset=['nu_cells'])

    return df



def apply_filters(df, filters):
    """Filter dataframe by exact values or ranges."""
    for key, val in filters.items():
        if key not in df.columns:
            raise KeyError(f"Column '{key}' not found in data.")
        if isinstance(val, tuple) and len(val) == 2:
            df = df[(df[key] >= val[0]) & (df[key] <= val[1])]
        else:
            df = df[df[key] == val]
    return df


def plot_datasets(
    datasets,
    xvar='DOF',
    output_path='plot.png',
    markers_only=False,
    use_latex=True,
    scale_x_to_millions=False,
    scale_E_to_GPa=False,
    reduce_xticks=False,
    figsize=(10, 7),
    xlabel_fontsize=24,
    ylabel_fontsize=24,
    title_fontsize=24,
    tick_fontsize=22,
    legend_fontsize=22,
    bold_text=False,
    legend_outside=False,
    log_x=False,    # 👈 NEW OPTION
    log_y=False     # 👈 NEW OPTION
):
    """
    Plot Exx, Eyy, Ezz, Emax, Emin vs xvar for one or more datasets.
    Dataset 1 = red tones, Dataset 2 = blue tones.
    Distinguish E-variables by different markers.
    Save PNG, PGF, and PDF outputs.
    """

    # --- Matplotlib / LaTeX setup ---
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "pgf.rcfonts": False,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif"
        })

    yvars = ['Exx', 'Eyy', 'Ezz', 'Emax', 'Emin']
    latex_labels = {
        'Exx': r"$E^*_{xx}$",
        'Eyy': r"$E^*_{yy}$",
        'Ezz': r"$E^*_{zz}$",
        'Emax': r"$E^*_{\max}$",
        'Emin': r"$E^*_{\min}$"
    }

    colorsets = [
        ['darkred', 'red'],          # dataset 1 colors
        ['navy', 'deepskyblue']      # dataset 2 colors
    ]
    markers = ['o', 's', '^', 'D', 'v']

    plt.figure(figsize=figsize)

    scale_factor_x = 1e6 if scale_x_to_millions else 1.0
    scale_factor_y = 1e3 if scale_E_to_GPa else 1.0

    # --- Plot each dataset ---
    for i, (df, label) in enumerate(datasets):
        if xvar not in df.columns:
            raise KeyError(f"xvar '{xvar}' not found in data.")

        df[xvar] = pd.to_numeric(df[xvar], errors='coerce')
        xdata = df[xvar] / scale_factor_x

        color_base = colorsets[i % len(colorsets)]
        for j, y in enumerate(yvars):
            if y not in df.columns:
                continue
            df[y] = pd.to_numeric(df[y], errors='coerce')
            ydata = df[y] / scale_factor_y

            color = color_base[j % len(color_base)]
            marker = markers[j % len(markers)]
            line_style = '' if markers_only else '-'
            legend_label = f"{label} - {latex_labels[y] if use_latex else y}"

            plt.plot(
                xdata,
                ydata,
                marker=marker,
                linestyle=line_style,
                color=color,
                label=legend_label,
                markersize=6,
                linewidth=2.0 if not markers_only else 0
            )

    # --- Axis labels ---
    if use_latex:
        xlabel = (r"$\mathrm{Number\ of\ Elements\ (millions)}$"
                  if scale_x_to_millions else r"$\mathrm{Number\ of\ Elements}$")
        ylabel = r"$E^*\ \mathrm{[GPa]}$" if scale_E_to_GPa else r"$E^*$"

    else:
        xlabel = f"{xvar} (millions)" if scale_x_to_millions else xvar
        ylabel = "E* [GPa]" if scale_E_to_GPa else "E*"


    plt.xlabel(xlabel, fontsize=xlabel_fontsize, fontweight='bold' if bold_text else 'normal')
    plt.ylabel(ylabel, fontsize=ylabel_fontsize, fontweight='bold' if bold_text else 'normal')

    # --- Tick formatting ---
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if bold_text:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    # --- Apply logarithmic scales if requested ---
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # --- Legend ---
    if legend_outside:
        plt.legend(
            prop={'size': legend_fontsize, 'weight': 'normal'},
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )
        plt.tight_layout(rect=[0, 0, 0.82, 1])  # Make room for legend
    else:
        plt.legend(prop={'size': legend_fontsize, 'weight': 'normal'})
        plt.tight_layout()

    # --- Reduce xticks if requested ---
    if reduce_xticks and not log_x:
        xticks = ax.get_xticks()
        if len(xticks) > 6:
            ax.set_xticks(np.linspace(min(xticks), max(xticks), 5))

    # --- Grid ---
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

    # --- Save to PNG, PGF, and PDF ---
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pgf'))
    plt.savefig(output_path.replace('.png', '.pdf'))
    plt.close()

    print(f"✅ Plot saved as {output_path}, {output_path.replace('.png', '.pgf')}, and {output_path.replace('.png', '.pdf')}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data.txt")

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return

    df = load_data(data_file)

    filters1 = {'deg': 1, 'atol': 0.08, 'nu_cells': (0.0,4000000.0)}
    filters2 = {'deg': 2, 'atol': 0.08, 'nu_cells':(0.0,4000000.0)}

    df1 = apply_filters(df, filters1)
    df2 = apply_filters(df, filters2)

    xvar = 'nu_cells'
    
    #xvar = 'DOF'

    # --- Plot styling options ---
    markers_only = True
    use_latex = True
    scale_x_to_millions = True
    scale_E_to_GPa = True
    reduce_xticks = True
    bold_text = False
    legend_outside = True

    # --- New logarithmic scaling options ---
    log_x = False   # 👈 Change to True for log-scale x-axis
    log_y = False   # 👈 Change to True for log-scale y-axis

    figsize = (10, 7)
    xlabel_fontsize = 24
    ylabel_fontsize = 24
    title_fontsize = 24
    tick_fontsize = 22
    legend_fontsize = 22

    datasets = []
    if not df1.empty:
        datasets.append((df1, "linear"))
    if not df2.empty:
        datasets.append((df2, "quadratic"))

    if not datasets:
        print("⚠️ No data matched the filter conditions.")
        return

    output_file = os.path.join(script_dir, f"plot_{xvar}.png")

    plot_datasets(
        datasets,
        xvar=xvar,
        output_path=output_file,
        markers_only=markers_only,
        use_latex=use_latex,
        scale_x_to_millions=scale_x_to_millions,
        scale_E_to_GPa=scale_E_to_GPa,
        reduce_xticks=reduce_xticks,
        figsize=figsize,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        title_fontsize=title_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        bold_text=bold_text,
        legend_outside=legend_outside,
        log_x=log_x,        # 👈 NEW
        log_y=log_y         # 👈 NEW
    )


if __name__ == "__main__":
    main()
