def convert_array_to_tif(data_array, filename, meta=None):
    """
    Function to convert 2D array into .tif data
    Args:
        data_array: 2D array, float value
        meta: meta of the tif as georeference for creating the .tif
        filename: output filename.tif
    Return: None, this function will not return any value
    """
    #import modul rasterio and affine
    from affine import Affine
    import rasterio
    from rasterio.crs import CRS

    #check if meta is provided or not
    if not meta:
        meta ={'driver': 'GTiff',
               'dtype': 'float32',
               'nodata': -9999.0,
               'width': 1680,
               'height': 1621,
               'count': 1,
               'crs': CRS.from_epsg(32750),
               'transform': Affine(5.0, 0.0, 815899.0,0.0, -5.0, 9902502.0968)}
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data_array, 1)
