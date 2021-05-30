import exifread
import json
import os

def read_GPS(img_name):
    GPS = {}
    img = open(img_name, 'rb')
    contents = exifread.process_file(img)
    
    lon = contents['GPS GPSLongitude'].printable
    lon = lon[1:-1].replace(" ", "").replace("/", ",").split(",")
    if len(lon) == 3:
        lon = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / 3600
    else:
        lon = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / float(lon[3]) / 3600

    GPS['longitude'] = lon

    lat = contents['GPS GPSLatitude'].printable
    lat = lat[1:-1].replace(" ", "").replace("/", ",").split(",")
    if len(lat) == 3:
        lat = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / 3600
    else:
        lat = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / float(lat[3]) / 3600
    
    GPS['latitude'] = lat
    
    return GPS

def main():
    img_info = {}
    input_dir = '/GPFS/data/haoningwu/HFNet/SJTU_Landmarks/db/'
    data = os.listdir(input_dir)
    for item in data:
        img = input_dir + item
        GPS = read_GPS(img)
        img_info[item] = GPS
    print(img_info)
    with open('db.json', 'w') as f:
        json.dump(img_info, f)

    return
if __name__ == '__main__' :
    main()