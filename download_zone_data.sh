#!/bin/bash
wget --directory-prefix ./data/zone_data https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
unzip ./data/zone_data/taxi_zones.zip -d ./data/zone_data
rm ./data/zone_data/taxi_zones.zip