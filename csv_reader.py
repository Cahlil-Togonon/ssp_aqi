from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point

dataset_path = "./dataset"
csv_filenames = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
print(csv_filenames)
sensor_cnt = len(csv_filenames)

# dict to assign all sensor AQI data to hourly date timestamps
data = {}
for csv_filename in csv_filenames:
    with open(csv_filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(filter(lambda row: row[0]!='#', csv_file))
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if row["date"] not in data:
                    data[row["date"]] = []
                data[row["date"]].append(float(row["median"]))
            line_count += 1

# hardcoded sensor locations (data from waqi api)
sensor_locations = {
    "wackwack_mandaluyong": [121.05573,14.591224],
    "forbestown_taguig": [121.043945,14.550762],
    "serendra_taguig": [121.05481,14.55091],
    "calzada_taguig": [121.07563,14.536089],
    "multinational_paranaque": [121.00152,14.487228]
}
sensor_names = [f.strip("./dataset").strip(".csv").strip("\\") for f in csv_filenames]
print(sensor_names)
sensor_X = []
sensor_Y = []
for sensor in sensor_names:
    sensor_X.append(sensor_locations[sensor][0])
    sensor_Y.append(sensor_locations[sensor][1])

# count all date timestamps with all sensor data present
total_dataset_cnt = 0
for key in data:
    if len(data[key]) == sensor_cnt:
        total_dataset_cnt += 1
print(total_dataset_cnt)

# generate dataframe and shapefile from date and AQI levels for interpolation
limit = total_dataset_cnt
for key in data:
    if limit == 0: break
    limit -= 1
    if len(data[key]) == sensor_cnt:
        df = pd.DataFrame({'Sensor Name':sensor_names,'X':sensor_X,'Y':sensor_Y,'US AQI':data[key]})
        geometry = [Point(xy) for xy in zip(df.X, df.Y)]
        df = df.drop(['X', 'Y'], axis=1)
        gdf = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
        gdf.to_file("./shapefiles/"+key.strip(":00:.000Z")+".shp")