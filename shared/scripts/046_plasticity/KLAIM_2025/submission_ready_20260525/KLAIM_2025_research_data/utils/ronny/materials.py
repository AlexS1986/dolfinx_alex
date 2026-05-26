materials_dict = {#rho in [kg/m^3], Emod in [MPa], nu, GIc in [MJ/m^2], KIc in [MPa*m^0.5], sigma_y in [MPa]
    "-": {"name": "Void", "rho": 0.0, "Emod": 0.0, "nu": 0.0, "GIc": 0.0, "KIc": 0.0, "sigma_y": 0.0},
    "Al2O3": {"name": "Aluminum Oxide", "rho": 3960.0, "Emod": 370.0e3, "nu": 0.22, "GIc": 4.0e-5, "KIc": 4.0, "sigma_y": 300.0},
    "TiN": {"name": "Titanium Nitride", "rho": 5220.0, "Emod": 600.0e3, "nu": 0.25, "GIc": 1.6e-5, "KIc": 2.7, "sigma_y": 300.0},
    "TiN_mod": {"name": "Titanium Nitride", "rho": 5220.0, "Emod": 600.0e3, "nu": 0.25, "GIc": 5.0e-4, "KIc": 2.7, "sigma_y": 300.0}, #"GIc": 1.0e-3
    "MgO": {"name": "Magnesium Oxide", "rho": 3560.0, "Emod": 300.0e3, "nu": 0.36, "GIc": 2.1e-5, "KIc": 2.7, "sigma_y": 120.0},
    #"CaO": {"name": "Calcium Oxide", "rho": 3340.0, "Emod": 300.0e3, "nu": 0.36, "GIc": 2.1e-5, "KIc": 2.7, "sigma_y": 120.0},
    "MnS": {"name": "Manganese Sulfide", "rho": 4000.0, "Emod": 120.0e3, "nu": 0.24, "GIc": 2.0e-6, "KIc": 0.5, "sigma_y": 0.0}, # GIc from K²(1-nu²)/E, K estimated from reports that it is much lower in coparison to steel's. 
    "fantasy": {"name": "ultra-mega material", "rho": 0.0, "Emod": 700.0e3, "nu": 0.19, "GIc": 3.0e-2, "KIc": 0.0, "sigma_y": 0.0},
    "100Cr6": {"rho": 7800.0, "Emod": 210.0e3, "nu": 0.3, "GIc": 0.0039, "KIc": 30.0, "sigma_y": 1000.0} #GIc calculated as K²(1-nu²)/E!
}

#100Cr6: #https://matweb.com/search/datasheet.aspx?MatGUID=d40ea0410a08497e8dee3f83364f2c76
#Al2=3: #https://www.matweb.com/search/datasheet.aspx?MatGUID=c8c56ad547ae4cfabad15977bfb537f1, https://pubs.aip.org/aip/jap/article/109/8/084305/930201/Critical-tensile-and-compressive-strains-for
#TiN: #https://asia.matweb.com/search/SpecificMaterialPrint.asp?bassnum=bntin0, fracture https://www.sciencedirect.com/science/article/pii/S025789722100921X
#MgO: https://www.azom.com/properties.aspx?ArticleID=54
#CaO: https://www.matweb.com/search/DataSheet.aspx?MatGUID=225c134dec664c4a9934e76eed06f2a5
#MnS: https://www.sciencedirect.com/science/article/pii/S0360319921022473, https://www.pveducation.org/pvcdrom/materials/mns no values reported for fracture toughness!

