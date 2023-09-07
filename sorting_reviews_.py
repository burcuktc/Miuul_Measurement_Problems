# SORTING REVIEWS

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# Up-Down Diff Score = (up ratings) − (down ratings)

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

#up-down diff. score kullanılması çok doğru bir yöntem dğeildir. Farka bakıldığında rewiev 2 seçilebilir, oran olarak review 1dir.
# Score = Average rating = (up ratings) / (all ratings)
###################################################

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

#bu yöntem de doğru olmaz çünkü bu yöntem de frekans bilgisini dikkate almamış oldu.

# Wilson Lower Bound Score
#İkili interactionlar barındıran herhangi bir item, product ya da review'ı skorlama imkanı sağlar.
#Bernouilli parametresi p için bir güven aralığı hesaplar bu güven aralığının alt sınırını WLB Skor olarak kabul eder.
#Bernouilli bir olasılık dağılımıdır ikili olayların olasılığını hesaplamak için kullanılır.
# 600-400
# 0.6
# 0.5 0.7
# 0.5



def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400) #0.5693094295142663
wilson_lower_bound(5500, 4500) #0.5402319557715324

wilson_lower_bound(2, 0) #0.3423802275066531
wilson_lower_bound(100, 1) #0.9460328420055449

# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})
#veri setinde normalde satırlarda yorumlar var. bu yourmların faydalı bulunma ve bulunmama sayıları da değişkenlerdedir. up- down
#önceki tüm yöntemleri getirelim
# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)



comments.sort_values("wilson_lower_bound", ascending=False)
#    up  down  score_pos_neg_diff  score_average_rating  wilson_lower_bound
#11  147     2                 145               0.98658             0.95238
#12   61     1                  60               0.98387             0.91413
#1    70     2                  68               0.97222             0.90426
#21   68     2                  66               0.97143             0.90168
#18   54     2                  52               0.96429             0.87881
#15   40     1                  39               0.97561             0.87405
#13   30     1                  29               0.96774             0.83806
#16   37     2                  35               0.94872             0.83114
#19   18     0                  18               1.00000             0.82412
#17   61     6                  55               0.91045             0.81807
#0    15     0                  15               1.00000             0.79612
#9    52     8                  44               0.86667             0.75835
#7    37     5                  32               0.88095             0.75000
#14   23     5                  18               0.82143             0.64409
#2    14     2                  12               0.87500             0.63977
#20   12     2                  10               0.85714             0.60059
#10   28    12                  16               0.70000             0.54570
#5     5     2                   3               0.71429             0.35893
#8    21    23                  -2               0.47727             0.33755
#6     8     6                   2               0.57143             0.32591
#3     4     2                   2               0.66667             0.29999
#4     2    15                 -13               0.11765             0.03288


