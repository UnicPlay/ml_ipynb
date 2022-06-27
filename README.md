# ml_ipynb
import pandas as pd
import numpy as np
pd.set_option("max_rows", 100)

# 1.1 Парсинг данных https://github.com/owid/covid-19-data/tree/master/public/data
df = pd.read_csv("owid-covid-data.csv", sep = ',')
# обоснование добавления данных туризма - readme
df2 = pd.read_csv("tourists.csv", sep = ',')
df2.rename(columns = {'Country Code':'iso_code'}, inplace = True)
df2.drop(df2.columns.difference(['iso_code','2018']), 1, inplace=True)
#df['tests_units'] = df['tests_units'].fillna('no data')
df_count = np.max(df.count())
# добавить текстовое описание в файл readme.txt!
print ("{:<50} {:<10} {:<10}".format('feature',  '# of nan', '% of nan'))
for column in df.columns:
    print("{:<50} {:<10} {:<10}".format(column, 
                                        df[column].isnull().sum(),
                                        round(100 * df[column].isnull().sum() / df_count)))
print("Countries with no information about tourists: ",df2['2018'].isnull().sum(), " from ", np.max(df2.count()))


#Оставить строчки с датами от 01 сентября 2021 и 200 дней после (~март-апрель 2022)
df['date']= ((pd.to_datetime(df['date']) - pd.to_datetime('2021-08-01')) / np.timedelta64(1, 'D')).astype(int)
df = df.drop(df[df['date'] < 0].index)
df = df.drop(df[df['date'] > 200].index)
# Удалить все агрегированные данные по континентам
df = df[df['continent'].notna()]
# Оставить только те атрибуты, для которых количество nan меньше 30%
df_count = np.max(df.count())
df = df.loc[:, df.isnull().sum(axis=0) < 0.3 * df_count]
print ("{:<50} {:<10} {:<10}".format('feature',  '# of nan', '% of nan'))
for column in df.columns:
    print("{:<50} {:<10} {:<10}".format(column, 
                                        df[column].isnull().sum(),
                                        round(100 * df[column].isnull().sum() / df_count)))


# 1.2 Предобработка данных и выделение значимых атрибутов
columns_to_delete = ['continent', 'location',
                 'total_cases', 'new_cases_per_million', 'new_cases_smoothed',
                 'total_deaths', 'new_deaths','new_deaths_smoothed', 'new_deaths_per_million',
                 'icu_patients', 'hosp_patients', 'weekly_icu_admissions',
                 'total_tests', 'new_tests', 'new_tests_per_thousand', 'new_tests_smoothed', 'positive_rate', 'tests_per_case', 'tests_units',
                 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
                 'new_vaccinations', 'new_vaccinations_smoothed',
                 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
                 'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred','new_people_vaccinated_smoothed',
                 'population_density', 'reproduction_rate',
                 'median_age', 'aged_65_older', 'aged_70_older',
                 'gdp_per_capita', 'extreme_poverty', 'hospital_beds_per_thousand',
                 'cardiovasc_death_rate', 'diabetes_prevalence',
                 'female_smokers', 'male_smokers',
                 'life_expectancy', 'human_development_index',
                 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative', 'excess_mortality']
df = df[df.columns.difference(columns_to_delete)]
df['immunity'] = (df['total_cases_per_million'] - df['total_deaths_per_million']) / 1000000 + 2 * df['new_people_vaccinated_smoothed_per_hundred']
df['month_cases_per_million'] = df['total_cases_per_million'] - np.roll(df['total_cases_per_million'], 30)
df['month_deaths_per_million'] = df['total_deaths_per_million'] - np.roll(df['total_deaths_per_million'], 30)
df.drop(columns=['total_cases_per_million', 'total_deaths_per_million'], inplace=True)
df['Rt4'] = 0.0

df = df.sort_values(by=['iso_code', 'date'])
data = df.to_numpy()
u, left_border = np.unique(data[:, df.columns.get_loc('iso_code')], return_index=True)
counties_number = u.size
right_border = np.roll(left_border, -1) 
right_border[-1] = data.shape[0]
for country in range(counties_number):
    for i in range(30):
        df.iloc[left_border[country] + i, df.columns.get_loc('month_cases_per_million')] = 0
        df.iloc[left_border[country] + i, df.columns.get_loc('month_deaths_per_million')] = 0
data = df.to_numpy()
    
df_count = np.max(df.count())
print ("{:<50} {:<10} {:<10}".format('feature',  '# of nan', '% of nan'))
for column in df.columns:
    print("{:<50} {:<10} {:<10}".format(column, 
                                        df[column].isnull().sum(),
                                        round(100 * df[column].isnull().sum() / df_count)))

# Количество пустых данных в %
#print(100 - 100 * df.count() / data.shape[0])



# 1.3 Описание структуры набора данных
# Информация об атрибутах находится на https://github.com/owid/covid-19-data/tree/master/public/data

import matplotlib.pyplot as plt

#fig, ax = plt.subplots()
#indices = list(range(left_border[2], right_border[2])) 
#line1 = ax.plot(data[indices, 0], data[indices, 2])
#line2 = ax.plot(data[indices, 0], data[indices, 3] * data[indices, df.columns.get_loc('population')] / 1000000)
#plt.show()

#fig, ax = plt.subplots()
#indices = list(range(left_border[2], right_border[2])) 
#line1 = ax.plot(data[indices, 0], data[indices, 5] * data[indices, df.columns.get_loc('population')] / 100)
#line2 = ax.plot(data[indices, 0], data[indices, 6] * data[indices, df.columns.get_loc('population')] / 1000000)
#plt.show()

print(df.describe())
figure, axis = plt.subplots((df.dtypes=='float64').sum(), 1)
figure.set_size_inches(15, 50)
i=0
for column in df.columns:
    if df.dtypes[column]=='float64':
        s=pd.Series(data[:, df.columns.get_loc(column)])
        axis[i].hist(s, bins=10, log=True)
        axis[i].set_title(column)
        i+=1
plt.show()


# 1.4 Формирование дополнительных атрибутов. Вычисление Rt по 4x4 дням согласно https://51.rospotrebnadzor.ru/content/809/51852/
column_Rt4 = df.columns.get_loc('Rt4')
column_new_cases = df.columns.get_loc('new_cases')
# параметер - интервал дней для оценки, по Роспотребнадзору 4
parameter = 7
for i in range(counties_number):
    sum_4_old = np.sum(data[left_border[i] : left_border[i] + parameter, column_new_cases]) 
    sum_4_new = np.sum(data[left_border[i] + parameter : left_border[i] + 2 * parameter, column_new_cases]) 
    for j in range(left_border[i] + 2 * parameter, right_border[i]):
        if sum_4_old > 0:
            data[j, column_Rt4] = sum_4_new / sum_4_old
        sum_4_old = sum_4_old - data[j - 2 * parameter, column_new_cases] + data[j - parameter, column_new_cases]
        if np.isnan(sum_4_old):
            sum_4_old = np.nansum(data[j - 2 * parameter + 1 : j - parameter + 1, column_new_cases]) 
        sum_4_new = sum_4_new - data[j - parameter, column_new_cases] + data[j, column_new_cases]
        if np.isnan(sum_4_new):
            sum_4_new = np.nansum(data[j - parameter + 1 : j + 1, column_new_cases]) 

fig, ax = plt.subplots()
indices = list(range(left_border[10], right_border[10]))
line1 = ax.plot(data[indices, 0], data[indices, column_new_cases])
line2 = ax.plot(data[indices, 0], data[indices, column_Rt4] * 10000)
plt.show()
# Анализ: На основании этого графика видно, что 4 - плохой параметр, 7 - получше, 14 - плохой


# 1.5 Кластеризация набора данных
# K-means       https://scikit-learn.org/stable/modules/clustering.html#k-means
# Mean shift    https://scikit-learn.org/stable/modules/clustering.html#mean-shift
# BIRCH         https://scikit-learn.org/stable/modules/clustering.html#birch
# Показатель оценки качества кластеризации: Компактность кластеров (Cluster Cohesion)

from sklearn.cluster import KMeans
model = Kmeans(n_clusters=3, random_state=0)
model.fit()



import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


print(__doc__)


# Generate sample data
n_samples = 1000
random_state = 0

X, _ = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)

# Number of cluster centers for KMeans and BisectingKMeans
n_clusters_list = [2, 3, 4, 5]

# Algorithms to compare
clustering_algorithms = {
    "K-Means": KMeans,
}

# Make subplots for each variant
fig, axs = plt.subplots(
    len(clustering_algorithms), len(n_clusters_list), figsize=(15, 5)
)

axs = axs.T

for j, n_clusters in enumerate(n_clusters_list):
    algo = KMeans(n_clusters=n_clusters, random_state=random_state)
    algo.fit(X)
    centers = algo.cluster_centers_

    axs[j].scatter(X[:, 0], X[:, 1], s=10, c=algo.labels_)
    axs[j].scatter(centers[:, 0], centers[:, 1], c="r", s=20)

    axs[j].set_title(f"{'KMeans'} : {n_clusters} clusters")


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


