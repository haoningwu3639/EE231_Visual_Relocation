import os
import folium
import json
import requests


def draw_map(locations, output_path, file_name):
    m = folium.Map(locations, zoom_start=25, width="100%", height="100%", zoom_control="False")
    folium.Marker(locations, popup="Location", icon = folium.Icon(color = "red")).add_to(m)
    m.save(os.path.join(output_path, file_name))

def geocoding(lat, lon):
    developer_key = 'Your Own Tencent Map Key'
    lat_lon = str(lat) + "," + str(lon)
    base = "https://apis.map.qq.com/ws/geocoder/v1/?key={}&location={}".format(developer_key, lat_lon) 

    response = requests.get(base)
    answer = response.json()
    result = answer['result']
    return result

def main():
    f = open('./db.json', 'r')
    data_json = json.load(f)
    f.close()

    loc = data_json['00407.jpg']
    lat, lon = loc['latitude'], loc['longitude']
    print(lat, lon)
    draw_map([lat, lon], './', 'map.html')
    geocoding(lat, lon)

if __name__ == '__main__':
    main()