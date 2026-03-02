To get the data on 6 hour basis::

    # install the requirements

    # change the data path and output path accordingly in the data_extract_6_hour.py

    ## DATA_PATH = ""

    ## OUTPUT_FILE = ""

    ## python data_extract_6_hour.py

    ## need to run it for train ,test, val by chaging here to get all the processed dataset ---(need to change in the line 84 and 85)##

To get the data on overall statistical data that has been used in the baseline paper(Linear Regression,Random Forest) ::

    ## in the parser add the correct data path
    # run python extract_all_statistics_feature.py --save_features

After that you can run to get the mean data from the generated CSV's with the correct path

    #python data_extract_mean.py
