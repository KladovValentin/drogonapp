import numpy as np
import ROOT
import pandas
import os
from array import array

def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, robust to constant inputs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.nan
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(np.nanmean(x * y) / (sx * sy))

def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """
    Rank data with average ranks for ties (1..N).
    Implements the common "average" method used in Spearman correlation.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")  # stable
    ranks = np.empty(n, dtype=float)
    a_sorted = a[order]

    i = 0
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        # average rank for ties; ranks are 1-based
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation = Pearson correlation of ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.nan
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return _pearson_corr(rx, ry)

def compute_mdc_feature_target_correlations(
    filepath: str,
    n_channels: int = 24,
    n_features: int = 9,
    avg_channels: int = 12,
    drop_nan: bool = True,
    ignore_zero_targets: bool = False,
):
    """
    Reads a whitespace-separated file with columns:
      run(int),  (n_channels*n_features features),
                (n_channels targets),
                (n_channels target_errors)

    Mapping: for each channel c, its feature block columns (c*n_features .. c*n_features+n_features-1)
             are correlated with target column c.

    Returns a dict with:
      pearson[ch, feat], spearman[ch, feat],
      pearson_abs_mean[feat], spearman_abs_mean[feat],
      pearson_mean[feat], spearman_mean[feat]
    """
    # Load full table
    arr = np.loadtxt(filepath)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    expected_cols = 1 + n_channels * n_features + n_channels + n_channels
    if arr.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected number of columns: got {arr.shape[1]}, expected {expected_cols} "
            f"(1 + {n_channels}*{n_features} + {n_channels} + {n_channels})."
        )

    run = arr[:, 0].astype(int)  # noqa: F841 (kept for potential debugging)
    off_feat = 1
    off_tgt  = off_feat + n_channels * n_features
    off_err  = off_tgt  + n_channels

    feats = arr[:, off_feat:off_tgt]                      # shape: (N, n_channels*n_features)
    tgts  = arr[:, off_tgt:off_err]                       # shape: (N, n_channels)
    errs  = arr[:, off_err:off_err + n_channels]          # shape: (N, n_channels)  # noqa: F841

    N = arr.shape[0]
    pearson = np.full((n_channels, n_features), np.nan, dtype=float)
    spearman = np.full((n_channels, n_features), np.nan, dtype=float)

    for ch in range(n_channels):
        y = tgts[:, ch]

        for f in range(n_features):
            x = feats[:, f * n_channels + ch]

            mask = np.ones(N, dtype=bool)
            if drop_nan:
                mask &= np.isfinite(x) & np.isfinite(y)
            if ignore_zero_targets:
                mask &= (y != 0.0)

            if mask.sum() < 3:
                continue

            pearson[ch, f] = _pearson_corr(x[mask], y[mask])
            spearman[ch, f] = _spearman_corr(x[mask], y[mask])

    # Aggregate across channels per feature
    channels_for_avg = min(avg_channels, n_channels)
    pearson_mean = np.nanmean(pearson[:channels_for_avg], axis=0)
    spearman_mean = np.nanmean(spearman[:channels_for_avg], axis=0)
    #pearson_abs_mean = np.nanmean(np.abs(pearson[:channels_for_avg]), axis=0)
    #spearman_abs_mean = np.nanmean(np.abs(spearman[:channels_for_avg]), axis=0)
    pearson_abs_mean = np.abs(np.nanmean(pearson[:channels_for_avg], axis=0))
    spearman_abs_mean = np.abs(np.nanmean(spearman[:channels_for_avg], axis=0))

    def _normalize_by_max_feature(arr: np.ndarray) -> np.ndarray:
        max_abs = np.nanmax(np.abs(arr))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            return arr
        return arr / max_abs

    pearson_mean_norm = _normalize_by_max_feature(pearson_mean)
    spearman_mean_norm = _normalize_by_max_feature(spearman_mean)
    pearson_abs_mean_norm = _normalize_by_max_feature(pearson_abs_mean)
    spearman_abs_mean_norm = _normalize_by_max_feature(spearman_abs_mean)

    return {
        "pearson": pearson,
        "spearman": spearman,
        "pearson_mean": pearson_mean,
        "spearman_mean": spearman_mean,
        "pearson_abs_mean": pearson_abs_mean,
        "spearman_abs_mean": spearman_abs_mean,
        "pearson_mean_norm": pearson_mean_norm,
        "spearman_mean_norm": spearman_mean_norm,
        "pearson_abs_mean_norm": pearson_abs_mean_norm,
        "spearman_abs_mean_norm": spearman_abs_mean_norm,
        "channels_used_for_avg": channels_for_avg,
    }

def plot_feature_correlations_root(
    corr_dict,
    feature_names=None,
    title_prefix="Feature–Target Correlations",
):
    """
    Creates ROOT TH1D bar-like plots overlaying Pearson and Spearman:
      - Signed correlations on the left pad
      - Absolute correlations on the right pad
    """
    pearson_mean = corr_dict.get("pearson_mean_norm", corr_dict["pearson_mean"])
    spearman_mean = corr_dict.get("spearman_mean_norm", corr_dict["spearman_mean"])
    pearson_abs_mean = corr_dict.get("pearson_abs_mean_norm", corr_dict["pearson_abs_mean"])
    spearman_abs_mean = corr_dict.get("spearman_abs_mean_norm", corr_dict["spearman_abs_mean"])

    n_features = len(pearson_mean)
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match n_features.")
    # Sort features by descending Pearson mean for clearer ranking in plots
    sort_order = np.argsort(np.nan_to_num(spearman_abs_mean+pearson_abs_mean, nan=-np.inf))[::-1]
    pearson_mean = pearson_mean[sort_order]
    spearman_mean = spearman_mean[sort_order]
    pearson_abs_mean = pearson_abs_mean[sort_order]
    spearman_abs_mean = spearman_abs_mean[sort_order]
    feature_names = [feature_names[i] for i in sort_order]

    def _make_hist(name, vals, title, y_min, y_max):
        h = ROOT.TH1D(name, title, n_features, 0.5, n_features + 0.5)
        h.SetStats(0)
        h.GetYaxis().SetRangeUser(y_min, y_max)
        h.GetXaxis().SetTitle("Feature")
        h.GetYaxis().SetTitle("Correlation")
        for i, v in enumerate(vals, start=1):
            h.SetBinContent(i, float(v) if np.isfinite(v) else 0.0)
            h.GetXaxis().SetBinLabel(i, feature_names[i - 1])
        h.GetXaxis().LabelsOption("v")
        return h

    c = ROOT.TCanvas("c_corr", "corr", 1800, 1000)
    c.Divide(2, 1)

    # Signed correlations
    c.cd(1)
    ROOT.gPad.SetGridy()
    h_p_signed = _make_hist("h_p_signed", pearson_mean, "", -1.1, 1.1)
    h_s_signed = _make_hist("h_s_signed", spearman_mean, "", -1.1, 1.1)
    h_p_signed.SetLineColor(ROOT.kBlue)
    h_p_signed.SetFillColor(ROOT.kBlue)
    h_p_signed.SetLineWidth(2)
    h_p_signed.SetFillStyle(3004)
    h_p_signed.GetXaxis().SetTitleOffset(1.25)
    h_p_signed.GetYaxis().SetTitleOffset(1.0)

    h_s_signed.SetLineColor(ROOT.kBlack)
    h_s_signed.SetFillColor(ROOT.kBlack)
    h_s_signed.SetLineWidth(2)
    h_s_signed.SetFillStyle(3005)
    h_p_signed.Draw("HIST")
    h_s_signed.Draw("HIST SAME")
    leg_signed = ROOT.TLegend(0.6, 0.75, 0.9, 0.9)
    leg_signed.AddEntry(h_p_signed, "Pearson signed", "f")
    leg_signed.AddEntry(h_s_signed, "Spearman signed", "f")
    leg_signed.Draw()

    # Absolute correlations
    c.cd(2)
    ROOT.gPad.SetGridy()
    h_p_abs = _make_hist("h_p_abs", pearson_abs_mean, "", 0.0, 1.1)
    h_s_abs = _make_hist("h_s_abs", spearman_abs_mean, "", 0.0, 1.1)
    h_p_abs.SetLineColor(ROOT.kBlue)
    h_p_abs.SetFillColor(ROOT.kBlue)
    h_p_abs.SetLineWidth(2)
    h_p_abs.SetFillStyle(3004)
    h_p_abs.GetXaxis().SetTitleOffset(1.25)
    h_p_abs.GetYaxis().SetTitleOffset(1.0)

    h_s_abs.SetLineColor(ROOT.kBlack)
    h_s_abs.SetFillColor(ROOT.kBlack)
    h_s_abs.SetLineWidth(2)
    h_s_abs.SetFillStyle(3005)
    h_p_abs.Draw("HIST")
    h_s_abs.Draw("HIST SAME")
    leg_abs = ROOT.TLegend(0.6, 0.75, 0.9, 0.9)
    leg_abs.AddEntry(h_p_abs, "Pearson abs", "f")
    leg_abs.AddEntry(h_s_abs, "Spearman abs", "f")
    leg_abs.Draw()

    c.SaveAs("corr_summary.pdf")
    c.Update()
    return c


def plotCorrelations():
    corr = compute_mdc_feature_target_correlations(
         "serverData/nn_input/outNNFitTargetBeam24_9.dat",
         n_channels=24,
         n_features=9,
         avg_channels=12,
         drop_nan=True,
         ignore_zero_targets=False,
    )
    featuresNames = ["P","HV","CO2","#DeltaP","H2O","Dew","T","N2","O2"]
    return plot_feature_correlations_root(corr, feature_names=featuresNames)

    
# If you want the full per-channel matrix for a given feature as a heatmap:
def plot_channel_feature_heatmap_root(corr_dict, which="pearson"):
    """
    which: 'pearson' or 'spearman'
    Produces a TH2D with channels on Y and feature index on X.
    """
    mat = corr_dict[which]
    n_channels, n_features = mat.shape
    h2 = ROOT.TH2D(
        f"h2_{which}",
        f"{which} correlation;feature;channel",
        n_features, 0.5, n_features + 0.5,
        n_channels, 0.5, n_channels + 0.5,
    )
    h2.SetStats(0)
    for ch in range(n_channels):
        for f in range(n_features):
            v = mat[ch, f]
            if np.isfinite(v):
                h2.SetBinContent(f + 1, ch + 1, float(v))
    c = ROOT.TCanvas(f"c2_{which}", f"{which} heatmap", 1200, 800)
    h2.Draw("COLZ TEXT")
    c.Update()
    return c




def compareInitialDistributions(aggregate_all: bool = False):
    mainPath = "/home/localadmin_jmesschendorp/gsiWorkFiles/realTimeCalibrations/backend/serverData/"
    pathFP = mainPath + 'function_prediction/'

    dftSim = pandas.read_parquet(pathFP + 'simuNormalized.parquet')
    dftExp = pandas.read_parquet(pathFP + 'tesuNormalized.parquet')

    print(dftSim)
    #selectionExp = (dftExp['beta']>1)
    #dftExp = dftExp.loc[selectionExp].copy().reset_index()
    #selectionSim = (dftSim['beta']>1)
    #dftSim = dftSim.loc[selectionSim].copy().reset_index()
    #dftSim = dftSim.loc[dftSim['pid'] == 3]
    #dftExp = dftExp.loc[dftExp['pid'] == 4]

    inputsLength = len(dftSim.columns)-1
    class_histS = [dftSim[(list(dftSim.columns)[i])].to_numpy() for i in range(inputsLength)]
    class_histE = [dftExp[(list(dftExp.columns)[i])].to_numpy() for i in range(inputsLength)]

    lowBord = [np.mean(class_histS[i]) - 4 * np.std(class_histS[i]) for i in range(inputsLength)]
    upBord = [np.mean(class_histS[i]) + 4 * np.std(class_histS[i]) + 8*np.std(class_histS[i])/200 for i in range(inputsLength)]
    step = [8*np.std(class_histS[i])/200 for i in range(inputsLength)]
    print(lowBord[0])
    print(upBord[0])
    print(step[0])
    #bins = [np.arange(lowBord[i], upBord[i], step[i]) for i in range(inputsLength)]

    # Create a 3x3 grid of subplots
    #fig, axes = plt.subplots(3, 4, figsize=(10, 8))

    c = ROOT.TCanvas(f"c_comp_{0}", "aaaa", 1100, 900)
    canvases = []

    i = 1
    if aggregate_all:
        title = "All Columns"
        data_sim = np.concatenate(class_histS)
        data_exp = np.concatenate(class_histE)
    else:
        title = str(list(dftSim.columns)[i + 1])
        data_sim = class_histS[i + 1]
        data_exp = class_histE[i + 1]

    h_sim = ROOT.TH1D(f"h_sim_{i}", title, 40, -3.5, 3.5)
    h_exp = ROOT.TH1D(f"h_exp_{i}", title, 40, -3.5, 3.5)

    h_sim.SetStats(0)
    h_exp.SetStats(0)

    for v in data_sim:
        h_sim.Fill(float(v))
    for v in data_exp:
        h_exp.Fill(float(v))

    #if h_sim.Integral() > 0.0:
    #    h_sim.Scale(1.0 / h_sim.Integral(), "width")
    #if h_exp.Integral() > 0.0:
    #    h_exp.Scale(1.0 / h_exp.Integral(), "width")

    h_sim.SetLineColor(ROOT.kBlue)
    h_sim.SetFillColor(ROOT.kBlue)
    h_sim.SetLineWidth(2)
    h_sim.SetFillStyle(3004)

    h_exp.SetLineColor(ROOT.kRed-2)
    h_exp.SetFillColor(ROOT.kRed-2)
    h_exp.SetLineWidth(2)
    h_exp.SetFillStyle(3005)
    

    h_sim.SetTitle(";Atmospheric pressure [a.u.];Runs count")
    h_sim.GetXaxis().SetTitleSize(0.045)
    h_sim.GetYaxis().SetTitleSize(0.045)
    h_sim.GetXaxis().SetTitleOffset(0.95)
    h_sim.GetYaxis().SetTitleOffset(1.0)
    y_max = max(h_sim.GetMaximum(), h_exp.GetMaximum()) * 1.1
    h_sim.SetMaximum(y_max)
    h_sim.Draw("HIST")
    h_exp.Draw("HIST SAME")

    leg = ROOT.TLegend(0.55, 0.7, 0.85, 0.85)
    leg.AddEntry(h_sim, "Train set", "lef")
    leg.AddEntry(h_exp, "Test set", "lef")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    #latex.DrawLatex(0.15, 0.85, "Module 1, Sector 1")

    c.Update()
    return c
        
    #fig.suptitle('"Feature" distributions, 9/128, both train AND validation dataset', fontsize=16)

    #plt.show()




def draw_hv_histogram(prediction_file, node_to_plot=0, total_hist_nodes=None, overlay_nodes=None):
    hv_table = np.loadtxt(prediction_file)
    if hv_table.ndim == 1:
        hv_table = hv_table.reshape(1, -1)

    if hv_table.shape[1] < 13:
        raise ValueError("Expected one run column plus 12 node columns in the HV prediction file.")

    n_nodes = 12
    histograms = []

    ROOT.TGaxis.SetMaxDigits(3)
    ROOT.gStyle.SetStripDecimals(True)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)

    def validate_nodes(nodes):
        validated = list(nodes)
        for node_idx in validated:
            if not 0 <= node_idx < n_nodes:
                raise ValueError(f"Node index {node_idx} is out of range [0, {n_nodes - 1}]")
        return validated

    def style_histogram(hist, line_color, fill_color=None, fillStyle=3004):
        hist.GetXaxis().SetTitle("High Voltage [kV]")
        hist.GetYaxis().SetTitle("Runs Count")
        hist.GetXaxis().SetTitleSize(0.042)
        hist.GetYaxis().SetTitleSize(0.042)
        hist.SetLineColor(line_color)
        hist.SetLineWidth(2)
        hist.SetFillColor(fill_color)
        hist.SetLineWidth(2)
        hist.SetFillStyle(fillStyle)
        #if fill_color is not None:
        #    hist.SetFillColorAlpha(fill_color, 0.35)

    for node_idx in range(n_nodes):
        node_values = hv_table[:, node_idx + 1].astype(float)
        mean_val = float(np.mean(node_values))
        std_val = float(np.std(node_values))
        hist_min = mean_val - 4.0 * std_val
        hist_max = mean_val + 4.0 * std_val

        hist = ROOT.TH1D(f"hv_node_{node_idx}", "", 80, hist_min, hist_max)
        style_histogram(hist, ROOT.kBlue + 1, ROOT.kAzure - 9)

        for value in node_values:
            hist.Fill(value)

        histograms.append(hist)

    canvas = ROOT.TCanvas("c_hv_distribution", "HV distribution", 1000, 1000)
    canvas.SetGridx()
    canvas.SetGridy()

    legend = ROOT.TLegend(0.12, 0.78, 0.42, 0.9)

    if total_hist_nodes is not None:
        selected_nodes = validate_nodes(total_hist_nodes)
        combined_values = hv_table[:, np.array(selected_nodes) + 1].reshape(-1)
        combined_values_original = hv_table[:, np.array(selected_nodes) + n_nodes + 1].reshape(-1)
        mean_val = float(np.mean(combined_values))
        std_val = float(np.std(combined_values))
        hist_min = mean_val - 4.0 * std_val
        hist_max = mean_val + 4.0 * std_val

        total_hist = ROOT.TH1D("hv_total_hist", "", 80, hist_min, hist_max)
        style_histogram(total_hist, ROOT.kBlue, ROOT.kBlue-2, 3004)
        total_hist_original = ROOT.TH1D("total_hist_original", "", 80, hist_min, hist_max)
        style_histogram(total_hist_original, ROOT.kRed, ROOT.kRed, 3001)
        for value in combined_values:
            total_hist.Fill(float(value))
        for value in combined_values_original:
            total_hist_original.Fill(float(value))

        total_hist.Draw("hist")
        total_hist.Fit("gaus", "Q")
        total_hist_original.Draw("hist same")
        fit_function = total_hist.GetFunction("gaus")
        if fit_function:
            fit_function.SetLineColor(ROOT.kRed + 1)
            fit_function.SetLineWidth(2)
            fit_function.Draw("same")
            legend.AddEntry(fit_function, "Gaussian fit", "l")
        legend.AddEntry(total_hist, "Optimal HV", "lf")
        legend.AddEntry(total_hist_original, "Initial HV", "lf")


    elif overlay_nodes is not None:
        selected_nodes = validate_nodes(overlay_nodes)
        selected_values = [hv_table[:, node_idx + 1].astype(float) for node_idx in selected_nodes]
        combined_values = np.concatenate(selected_values)
        hist_min = float(np.min(combined_values))
        hist_max = float(np.max(combined_values))
        color_sequence = [
            ROOT.kBlue + 1,
            ROOT.kRed + 1,
            ROOT.kGreen + 2,
            ROOT.kMagenta + 1,
            ROOT.kOrange + 7,
            ROOT.kCyan + 1,
        ]

        overlay_hists = []
        for draw_idx, node_idx in enumerate(selected_nodes):
            node_values = hv_table[:, node_idx + 1].astype(float)
            hist = ROOT.TH1D(f"hv_overlay_node_{node_idx}", "", 80, hist_min, hist_max)
            color = color_sequence[draw_idx % len(color_sequence)]
            style_histogram(hist, color)
            for value in node_values:
                hist.Fill(value)
            overlay_hists.append(hist)

        overlay_hists[0].Draw("hist")
        legend.AddEntry(overlay_hists[0], f"Optimal HV, node {selected_nodes[0]}", "l")
        for draw_idx in range(1, len(overlay_hists)):
            overlay_hists[draw_idx].Draw("hist same")
            legend.AddEntry(
                overlay_hists[draw_idx],
                f"Optimal HV, node {selected_nodes[draw_idx]}",
                "l",
            )

    else:
        if not 0 <= node_to_plot < n_nodes:
            raise ValueError(f"node_to_plot must be in [0, {n_nodes - 1}]")

        selected_hist = histograms[node_to_plot]
        selected_hist.Draw("hist")
        selected_hist.Fit("gaus", "Q")
        fit_function = selected_hist.GetFunction("gaus")
        if fit_function:
            fit_function.SetLineColor(ROOT.kRed + 1)
            fit_function.SetLineWidth(2)
            fit_function.Draw("same")
            legend.AddEntry(fit_function, "Gaussian fit", "l")
        legend.AddEntry(selected_hist, "Optimal HV", "lf")

    legend.Draw()
    canvas.Update()

    canvas.SaveAs("qaPlots/optimalHVtotal.pdf")
    canvas.SaveAs("qaPlots/optimalHVtotal.png")

    if not ROOT.gROOT.IsBatch():
        try:
            input("Press Enter to exit and close the plot window...")
        except KeyboardInterrupt:
            pass

    return histograms





# Example usage:
if __name__ == "__main__":
    #canvas = plotCorrelations()
    
    #canvases = compareInitialDistributions()
    
    draw_hv_histogram(
        "serverData/function_prediction/predicted/predicted_0HV.txt",
        node_to_plot=0,
        total_hist_nodes=[0,1,2,3,4,5]
    )
    
    if not ROOT.gROOT.IsBatch():
        try:
            input("Press Enter to exit and close the plot window...")
        except KeyboardInterrupt:
            pass
