# Uygulama: IMDB Movie Scoring & Sorting

import pandas as pd
import math
import scipy.stats as st
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df = pd.read_csv("C:/Users/asus/Desktop/miuul/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin)
df.head()
df = df[["title", "vote_average", "vote_count"]]
df.head()
df.shape
# Vote Average'a Göre Sıralama
df.sort_values("vote_average", ascending=False).head(20)
#mantıklı olmadı çünkü örneğin sadece 1 kişinin değerlendirdiği ve 10 puan verdiği film en yukarda çıktı. (vote_count =1)
#vote_count'u belirli değerden yukarda olanlara göre sıralama yapmak isteyebiliriz. Ama hangi değerden yukarda almanın mantıklı olması gerekir bilmiyoruz. Bunu çözmek için vote_count'a describe atıp çeyrek değerler veriyoruz...
#... yani %10, %25,... gibi oy sayılarının çeyrek değerlerini göster diyoruz.
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T #çeyrek değerlere göre filmlerin aldığı oy sayıları geldi
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)
#vote_count değişkenini standartlaştırmak istiyoruz:
from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])
# vote_average * vote_count
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values("average_count_score", ascending=False).head(20)

# IMDB Weighted Rating
#IMDB'nin 2015yılına kadar kullandığı genel bir formul vardır. 2 durumu göz önünde bulundurur.
# 1.C:kitlenin genel ortalaması (tüm filmlerin genel ortalaması) 2.m:sıralamaya girebilmek için gerekli olan minimum oy sayısı
#R: her bir filmin kendi puanını ifade eder. V: Oy sayısı
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)
# Bayesian Average Rating Score

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction



def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351]) #esaretin bedeli filmi kaç tane 1 yıldız almış, kaç tane 2 yıldız almış, ..., kaç tane 10 yıldız almış

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905]) #baba filmi kaç tane 1 yıldız almış, kaç tane 2 yıldız almış, ..., kaç tane 10 yıldız almış

df = pd.read_csv("datasets/imdb_ratings.csv") #veri setinde oyların puan dağılımı var
df = df.iloc[0:, 1:]


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.


#simple recommender örneği, basit tavsiye sistemi örneği