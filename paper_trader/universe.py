"""Universe of Canadian-listed ETFs the paper trader can pick from.

~111 of the most liquid Canadian ETFs across all major providers:
iShares, BMO, Vanguard, Horizons/Global X, CI, Hamilton, Purpose,
Mackenzie. Curated for liquidity so the paper trader's simulated
fills are realistic.
"""
from __future__ import annotations

UNIVERSE: list[str] = [
    # --- Broad Canadian equity ---
    "XIU.TO", "XIC.TO", "VCN.TO", "ZCN.TO", "HXT.TO", "XCS.TO", "XMD.TO",
    # --- US equity (CAD-hedged + unhedged + swap variants) ---
    "VFV.TO", "VUN.TO", "XSP.TO", "ZSP.TO", "HXS.TO", "ZUE.TO",
    "VUS.TO", "ZUQ.TO", "XUS.TO",
    # --- US Nasdaq / tech ---
    "ZQQ.TO", "XQQ.TO", "HXQ.TO", "QQC.TO",
    # --- International developed ---
    "XEF.TO", "XIN.TO", "VI.TO", "VIU.TO", "XAW.TO", "VXC.TO", "XWD.TO",
    # --- Emerging markets ---
    "XEC.TO", "ZEM.TO", "VEE.TO", "XEM.TO",
    # --- Canadian sectors ---
    "XEG.TO", "ZEO.TO", "ZEB.TO", "XFN.TO", "ZGI.TO", "ZUT.TO", "XUT.TO",
    "XMA.TO", "XGD.TO", "ZJG.TO", "XIT.TO", "XHC.TO", "XST.TO", "XCD.TO",
    "XRE.TO", "ZRE.TO",
    # --- Dividend / income ---
    "XDV.TO", "VDY.TO", "XEI.TO", "ZDV.TO", "CDZ.TO", "XHD.TO", "VIDY.TO",
    # --- Covered-call income ---
    "ZWB.TO", "ZWC.TO", "ZWU.TO", "ZWE.TO", "ZWH.TO", "ZWS.TO",
    "HMAX.TO", "UMAX.TO", "HDIV.TO",
    # --- Low volatility ---
    "ZLB.TO", "ZLU.TO", "ZLE.TO",
    # --- Fixed income ---
    "XBB.TO", "ZAG.TO", "VAB.TO", "XSB.TO", "VSB.TO",
    "XGB.TO", "ZGB.TO", "XCB.TO", "ZCS.TO",
    "XHY.TO", "ZHY.TO", "XSH.TO", "ZEF.TO", "XPF.TO",
    # --- All-in-one asset allocation ---
    "XEQT.TO", "VEQT.TO", "XGRO.TO", "VGRO.TO",
    "XBAL.TO", "VBAL.TO", "VCNS.TO", "XCNS.TO", "VRIF.TO",
    # --- Cash / HISA equivalents ---
    "PSA.TO", "CASH.TO", "ZST.TO",
    # --- Horizons single + leveraged ---
    "HXX.TO",
    "HOU.TO", "HOD.TO", "HNU.TO", "HND.TO",
    "HSU.TO", "HSD.TO", "HQU.TO", "HQD.TO",
    "HXU.TO", "HXD.TO",
    # --- Specialty / thematic ---
    "CGL.TO", "MNT.TO", "BTCC.TO", "ETHX.TO", "RBOT.TO",
]
