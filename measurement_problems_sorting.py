# Ürün Sıralama ( Sorting Products )
#Sıralama, sadece ürünler için değil birçok alanda karşımıza gelebilir. Örneğin iş ilanına başvuran adayların 1)mezuniyet puanı 2)yabancı dil puanı 3)mülakatta aldıkları puan'a ait belirli skorları olsun...
#... sıralamak için üç faktöre de ağırlık tanımlayıp bu ağırlıklara göre sıralama yapılabilir.

#Derecelendirmeye göre Sıralama (Sorting by Rating)
#Online eğitim platformundaki kursların sıralaması uygulaması:
# Uygulama: Kurs Sıralama
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv("C:/Users/asus/Desktop/miuul/product_sorting.csv")
df.shape
#verisetinde kurs adı, eğitmen isimleri, satın alma sayıları, kursun ort. puanı, kursun aldığı yorum sayısı ve bu yorumların puan cinsinden dağılımı (5 puan veren x kişi, 4 puan veren y kişi vs)
# Sorting by Rating
df.sort_values("rating", ascending=False).head(20)
#fakat bu sıralamada satın alma sayıları ve yorum sayıları dikkate alınmamış oluyor. Yapmamız gereken hem satın alma sayısını, hem puanı, hem yorum sayısını dikkate alarak sıralama yapmaktır.

# Sorting by Comment Count or Purchase Count
df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

# Sorting by Rating, Comment and Purchase (Derecelendirme, satın alma,yoruma göre sıralama)
#3 faktörü (rating, comment ve purchase) aynı anda göz önünde bulunduracağız. Bunun için değişkenleri ölçeklenebilir forma getireceğiz. Örneğin rating 1-5 arasındaki sayılardan oluşuyor. Satın alma ve yorum değişkenlerini de bu ölçeğe göre ayarlayalım.
from sklearn.preprocessing import MinMaxScaler #ölçeklendirmek için MinMaxScaler kullanılır.
#satın almayı ölçeklendirmek için:
#1) oluşturacak olduğum değişken 1-5 arasında olacak. feature range 1-5.
#2)daha sonra fit et
#3)dönüştürmeyi yap
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])
df.describe().T

#yorum sayısına göre ölçeklendirme
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

#Bu 3 değişkenin aynı cinsten oldu. Artık bunların ortalaması alınabilir, ağırlıkları alınabilir.
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

#kurs isminde sadece veri bilimi olanları getirmek için:
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# Bayesian Average Rating Score
#Ratingleri daha farklı açıdan hassaslaştırmak, sadece ratinge odaklanarak bir sıralama yapma.
#Rating için bir skor elde edeceğiz. Bu skor bir ürünün nihai ortalama puanı olarak da kullanılabilir, bir skor olarak da değerlendirilebilir.
#Nihayetinde bayesian average rating, rating ile bize ortalama değer verecek.
# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating
#bayesian average: puan dağılımı üzerinden ağırlıklı şekilde olasılıksal ortalama hesabı yapar.
import math
import scipy.stats as st
#bayesian average rating fonksiyonu, yapılacak işlemi bayesian metodla ağırlıklı ortalama hesabı olduğunu söyleyebilriz.
#Biz bir ortalama hesabı yapıyoruz  ama elimizde geçmişte elde ettiğimiz puanların dağılımı var yani geçmiş bilgi var geçmiş bilgiyi kullanarak geleceği kurmak için elimizde var olan değerler üzerinden tekrar bir rating hesabı yapacağız ama bu hesap olasılıksal olacak.
#fonksiyonda n girilecek olan yıldızların ve bu yıldızlara ait gözlenme frekanslarını ifade eder. Yani örneğin n 5 elemanlı, 1. elemanında 1 yıldızdan kaç tane var, 2. elemanında 2 yıldızdan kaç tane var bilgileri girildikten sonra hesaplama işlemi gerçekleştirilecek.

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


df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)
#eğer odağımız sadece kursa verilen puanlara göre sıralama yapmak ise bar score yöntemi kullanılabilir. Ama bu sıralamada kullanıcı yorumları gibi etkenler dikkate alınmamış olur

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)

# Hybrid Sorting: BAR Score + Diğer Faktorler

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler

#wss daha önce hesapladığımız ağırlıklı yöntem (3 etken hesaba katılarak), bar_score ise bayesian yöntem
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)
#ürün sıralamada;
#iş bilgisi açısından önemli olabilecek faktörler göz önünde bulundurulmalı.
#eğer birden fazla faktör varsa bu faktörlerin etkileri aynı anda göz önünde bulundurulmak üzere önce standartlaştırılmalı daha sonra etkilerin farkı varsa ağırlıklandırılmalı
#literatürdeki sitatiksel bazı yöntemler güvenilir de olsa bu yöntemleri tek başına kullanmak yerine iş bilgisi ile harmanlanacak şekilde birlikte kullanılmalıdır.
