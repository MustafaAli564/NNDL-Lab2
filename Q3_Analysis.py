import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations

def test(LR, LRL2, LRL1, NN, NNL2, NNL1):
    # Perform Friedman test
    stat, p_value = friedmanchisquare(LR, LRL2, LRL1, NN, NNL2, NNL1)
    print(f"Friedman test statistic: {stat}, p-value: {p_value}")

    # Perform pairwise Wilcoxon signed-rank test if Friedman test is significant
    wilcoxon_results = {}
    models = ["LR", "LRL2", "LRL1", "NN", "NNL2", "NNL1"]
    model_data = [LR, LRL2, LRL1, NN, NNL2, NNL1]

    if p_value < 0.05:  # If Friedman test is significant, conduct pairwise tests
        for (i, j) in combinations(range(len(models)), 2):
            try:
                w_stat, w_p = wilcoxon(model_data[i], model_data[j], zero_method='zsplit')
                wilcoxon_results[f"{models[i]} vs {models[j]}"] = (w_stat, w_p)
            except ValueError as e:
                wilcoxon_results[f"{models[i]} vs {models[j]}"] = str(e)

    # Print Wilcoxon results
    for comparison, (w_stat, w_p) in wilcoxon_results.items():
        print(f"{comparison}: Wilcoxon statistic = {w_stat}, p-value = {w_p}")

# Define AUC values for six models across 10 test cases

#%%
LR_df1 = [
            0.8281709758962765,
            0.824808042512887,
            0.8192568699803566,
            0.8286788013419567,
            0.8233667667988411,
            0.8200300325260588,
            0.822726941445903,
            0.820330999147379,
            0.8181240185642187,
            0.814068241170552,
        ]
LRL2_df1 = [0.8281708592575797,
0.824807031644182,
0.8192561183087554,
0.8286795659734129,
0.8233667667988409,
0.8200305248837922,
0.8227278484206751,
0.8203306493142527,
0.8181243036134327,
0.8140681375162924,
]
LRL1_df1 = [0.8281721552430985,
0.8248079777136109,
0.8192565200642663,
0.8286790605390605,
0.8233671166319674,
0.8200303434888377,
0.8227267341373837,
0.8203323207391897,
0.8181255863348962,
0.8140685132629837,
]
NN_df1 = [0.8294409509858794,
0.8264130492989263,
0.8192916283119753,
0.8308429546002276,
0.8241985727974555,
0.8206079503724136,
0.8226437070753945,
0.8195236555540446,
0.8190100681325927,
0.8130651724214106,
]
NNL2_df1 = [0.8038478030375723,
0.7985038197099594,
0.795849518248487,
0.805852070562368,
0.8010370673463103,
0.7969220321080731,
0.7988367141573774,
0.7929682440275226,
0.7930995480609494,
0.7877033915220274,
]
NNL1_df1 = [0.8054096081463368,
0.7999398818237461,
0.794060753675713,
0.8066544994362204,
0.7992639511021395,
0.7990727284282366,
0.7953046369538708,
0.7952626569787076,
0.7954628133541152,
0.7914700771854967,
]
test(LR_df1,LRL2_df1,LRL1_df1,NN_df1,NNL2_df1,NNL1_df1)

#%%
LR_df2 = [0.86170465337132,
0.8875830959164293,
0.8548195631528964,
0.8790360873694206,
0.9112060778727444,
0.9094254510921178,
0.899810066476733,
0.9037274453941121,
0.917028008906598,
0.8872431915910176,
]
LRL2_df2 = [0.86170465337132,
0.8875830959164293,
0.8548195631528964,
0.8790360873694206,
0.9112060778727444,
0.9094254510921178,
0.899810066476733,
0.9037274453941121,
0.917028008906598,
0.8872431915910176,
]
LRL1_df2 = [0.8618233618233618,
0.8884140550807217,
0.8538698955365622,
0.8782051282051282,
0.9117996201329535,
0.9095441595441595,
0.8998100664767331,
0.9052706552706553,
0.9160904722840736,
0.8872431915910176,
]
NN_df2 = [0.867165242165242,
0.8349952516619183,
0.8538698955365622,
0.8555318138651472,
0.8584995251661918,
0.9104938271604939,
0.8871082621082621,
0.9019468186134852,
0.9060119535919372,
0.8588150979455328,
]
NNL2_df2 = [0.8637226970560304,
0.7531457739791073,
0.8157644824311491,
0.9122744539411206,
0.8590930674264007,
0.907644824311491,
0.8881766381766382,
0.9033713200379867,
0.847767490917614,
0.8084089823220257,
]
NNL1_df2 = [0.8811728395061729,
0.8084045584045584,
0.7986704653371319,
0.8727445394112061,
0.8517331433998101,
0.8873456790123457,
0.8939933523266856,
0.8637226970560303,
0.8715574827141686,
0.8708791208791209,
]
test(LR_df2,LRL2_df2,LRL1_df2,NN_df2,NNL2_df2,NNL1_df2)

#%%
LR_df3 = [0.892128279883382,
0.9597667638483965,
0.9300291545189504,
0.8944606413994168,
0.9556851311953354,
0.9008746355685131,
0.892128279883382,
0.9467592592592593,
0.8912037037037037,
0.9196428571428572,
]
LRL2_df3 = [0.892128279883382,
0.9597667638483965,
0.9300291545189504,
0.8944606413994168,
0.9556851311953354,
0.9008746355685131,
0.892128279883382,
0.9467592592592593,
0.8912037037037037,
0.9196428571428572,
]
LRL1_df3 = [0.8915451895043732,
0.9574344023323615,
0.9300291545189504,
0.8962099125364431,
0.9545189504373178,
0.9008746355685132,
0.8932944606413994,
0.9456018518518519,
0.8929398148148149,
0.9208333333333334,
]
NN_df3 = [0.9043731778425655,
0.9107871720116618,
0.9346938775510204,
0.8909620991253645,
0.972594752186589,
0.8787172011661808,
0.9253644314868804,
0.9305555555555556,
0.8709490740740741,
0.9345238095238095,
]
NNL2_df3 =[0.9137026239067056,
0.9119533527696793,
0.9160349854227405,
0.8699708454810495,
0.9860058309037901,
0.8699708454810495,
0.9282798833819241,
0.912037037037037,
0.855324074074074,
0.9172619047619048,
]
NNL1_df3 = [0.9113702623906705,
0.8997084548104956,
0.9177842565597667,
0.8489795918367347,
0.9498542274052477,
0.8559766763848395,
0.9276967930029154,
0.8790509259259259,
0.8287037037037036,
0.881547619047619,
]
test(LR_df3,LRL2_df3,LRL1_df3,NN_df3,NNL2_df3,NNL1_df3)

#%%
LR_df4 = [0.9922077922077921,
0.9987012987012986,
0.996031746031746,
0.996031746031746,
0.9775132275132274,
0.9907407407407407,
1.0,
1.0,
0.996031746031746,
1.0,
]
LRL2_df4 = [0.9922077922077921,
0.9987012987012986,
0.996031746031746,
0.996031746031746,
0.9775132275132274,
0.9907407407407407,
1.0,
1.0,
0.996031746031746,
1.0,
]
LRL1_df4 = [0.9948051948051948,
0.9987012987012986,
0.9973544973544973,
0.9947089947089947,
0.9788359788359788,
0.9907407407407407,
1.0,
1.0,
0.996031746031746,
1.0,
]
NN_df4 = [0.9597402597402598,
0.9779220779220779,
0.9947089947089947,
0.9325396825396826,
0.906084656084656,
0.9775132275132276,
0.9497354497354498,
0.9920634920634921,
0.9801587301587302,
0.9945578231292517,
]
NNL2_df4 =[0.9805194805194805,
0.9727272727272727,
0.9880952380952381,
0.9537037037037037,
0.943121693121693,
0.9854497354497355,
0.9484126984126985,
0.9947089947089947,
0.9828042328042328,
0.9945578231292517,
]
NNL1_df4 = [1.0,
0.9818181818181819,
0.9775132275132274,
0.9682539682539683,
0.9457671957671957,
0.9814814814814815,
1.0,
0.9920634920634921,
0.9642857142857143,
1.0,
]
test(LR_df4,LRL2_df4,LRL1_df4,NN_df4,NNL2_df4,NNL1_df4)

#%%
LR_df5 = [0.6887950259366157,
0.7014648960592107,
0.6997503718801099,
0.6834991405123836,
0.7071777692230696,
0.6887372708005882,
0.6820007638442426,
0.7002899264403664,
0.6882780364508272,
0.6886796058981489,
]
LRL2_df5 = [0.6887957858726161,
0.7014651131837824,
0.6997497205063952,
0.6834993576369551,
0.707175272290497,
0.6887400934200182,
0.6819988097230989,
0.7002918805615101,
0.6882787963315564,
0.6886789543321752,
]
LRL1_df5 = [0.6887937231891865,
0.7014666330557829,
0.6997544972469691,
0.6835121679866756,
0.7071820031522146,
0.6887509496485946,
0.6820073861436743,
0.7002978514872273,
0.688288566226645,
0.6886768910399254,
]
NN_df5 = [0.9932105146482006,
0.9893796772703795,
0.9954649190744154,
0.9844371620863153,
0.9926086453359145,
0.9840943223878666,
0.9908315892802125,
0.9933228766139685,
0.9865436521083651,
0.9950139999808874,
]
NNL2_df5 =[0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
]
NNL1_df5 = [0.4935705071878003,
0.5,
0.5029821517088464,
0.5251735313856825,
0.4277718285645178,
0.48111157358644846,
0.5,
0.5180280160177139,
0.4974792585127267,
0.48611339159201894,
]
test(LR_df5,LRL2_df5,LRL1_df5,NN_df5,NNL2_df5,NNL1_df5)