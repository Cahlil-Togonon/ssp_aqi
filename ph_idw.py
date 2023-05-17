from pyidw import idw

def get_idw(date_time):
    idw.idw_interpolation(
        input_point_shapefile="./shapefiles_interpolated/"+date_time+".shp",
        extent_shapefile="./borders/MetroManila_Border.shp",
        column_name="US AQI",
        power=2,
        search_radious=5,
        output_resolution=100,
    )