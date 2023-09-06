#Measurement Problems (Ölçüm Problemleri)
#Veri bilimi alanında sık karşılaşılan bazı ölçme problemleri ele alınacaktır.
# Bir ürünü satın aldıran nedir?
#Kullanıcıların satın alma kararını etkileyen faktörlerden son zamanlarda en etkilisi social proof (sosyal ispat) kavramıdır.
#Ürün yorumları, puanları, toplumun ürüne ne tepkini verdiğini anlamak üzerinden ürünle ilgili kanaat elde edilmeye çalışılır.
#Topluluğun olumlu görüşünü bize kabul ettiren şey ise wisdom of crowd (kalabalıkların bilgeliği)'ne olan inançtır.
#Ürün puanlarının hesaplanması, ürünlerin sıralanması (satın alma sayısına göre mi, yoruma göre mi, aldığı puana göre mi gibi),ürün detay sayfalarındaki kullanıcıların yorumlarının sıralanması,...
# ...sayfa süreç ve etkileşim alanlarının tasarımları, özellik denemeleri, olası aksiyonların ve reaksiyonların  test edilmesi
#Yöntemler: rating products, sorting products, sorting reviews, AB Testing, Dynamic Pricing
#Rating Products: Olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlama
#Ürün puanlama uygulaması için gerçek bir veri setinden uygulama:

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

#Bir ürüne verilen puanlar üzerinden çeşitli değerlendirmeler yaparak en doğru puanın nasıl hesaplanabileceğini dair uygulama:
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("C:/Users/asus/Desktop/miuul/course_reviews.csv")
#veri seti kullanıcıların bir kursa verdikleri puanları ifade etmektedir.
df.head()
df.shape
#rating dağılımı
df["Rating"].value_counts()

#sorulan sorular
df["Questions Asked"].value_counts()

#sorulan soru kırılımında verilen puan (soru sayılarına göre ortalama puan örneğin 2 soru soran 80 kişi ort kaç puan vermiş)
df.groupby("Questions Asked").agg({'Questions Asked': "count",
                                   "Rating": 'mean'})

#Ortalama puan
df["Rating"].mean()
#sadece direk bu şekilde bir ortalama hesabı yapıldığında iligili ürün (veya bu örnekteki eğitim) ile ilgili müşteriler açısından son zamanlardaki memnuniyet trendini kaçırıyor olabiliriz.
#Örneğin 1 yıl önce ilk 3 ayda bu ürün ile ilgili memnuniyet yüksek olabilir ama bir yılın son 3 ayında ürünle ilgi ortaya bazı sorunlar çıkmış olabilir ve memnuniyet düşmüş olabilir.
#Güncel trendi ortalamaya daha iyi yansıtmak için puan zamanlarına göre ağırlıklı ortalama yapılabilir

# Time-Based Weighted Average
# Puan Zamanlarına Göre Ağırlıklı Ortalama
df.info()
#timestamp değişkeninin dtype'ı object olduğunu gördük datetime'a çevirmek için:

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

current_date = pd.to_datetime('2021-02-10 0:0:0')#bugünün tarihini 2021-02-10 yaptık:
#bugünün tarihinden tüm yorumların tarihini çıkarıp gün cinsinden ifade etmek için yani yorumların kaç gün önceden yapıldığını belirlemek için:
df["days"] = (current_date- df["Timestamp"]).dt.days

#verisetinde son 30 gün önce yapılan yorumlara erişmek için:
df[df["days"] <= 30]
#kaç tane varmış ?
df[df["days"] <= 30].count()

#son 30 günde yapılan yorumların ortalamasını almak istersek: loc(satırlardan bir index bilgisi verdik koşul olarak ve sadece "Rating" değişkeni gelsin)
df.loc[df["days"] <= 30,"Rating"].mean()
#30dan büyük ve 90dan küçük eşitlerin ortalaması
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
#90dan büyük ve 180den küçük eşit günler
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[(df["days"] > 180), "Rating"].mean()

#ağırlıklar
df.loc[df["days"] <= 30,"Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

#bunu fonksiyon olarak yazmak için:

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

# User-Based Weighted Average (Kullanıcı Temelli Ağırlıklı Ortalama)
#Tüm kullanıcıların verdiği puanlar aynı ağırlığa mı sahip olmalı
#Örneğin kursun %5'ini izleyen biri ile tamamını izleyen bir kişinin yorumu aynı ağırlığa mı sahip olmalı
#Bu örnekte kursun ilerleme durumuna göre ağırlıklar ile ilgili puanlama yapmak istiyoruz:
df.groupby("Progress").agg({'Rating': 'mean'})
df.loc[df["Progress"] < 10, "Rating"].mean() * 22/100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

# Weighted Rating
#Bu bölümde time based ve user based ile yaptığımız ağırlıklı ortalamaları bir araya getirerek tek fonk. kullanarak hesaplama yapacağız
def course_weighted_rating(dataframe,time_w=0.50, user_w=50)
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)* user_w/100
course_weighted_rating(df)
#time'dan gelen ağırlık daha önemliyse örneğin time_w=60, user_w=40 yapabiliriz.
course_weighted_rating(df, time_w=40, user_w=60)

#böylece zamana ve user quality değerlerine göre daha güven verici, daha hassas ortalama hesabını gerçekleştrimiş olduk.
#######
#ÜRÜN SIRALAMA
