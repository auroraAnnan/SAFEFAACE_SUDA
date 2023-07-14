def database_interface(driver, server, database, uid, pwd):
    return 'DRIVER={driver};SERVER={server};DATABASE={database};UID={uid};PWD={pwd}'.format(driver=driver,
                                                                                            server=server,
                                                                                            database=database, uid=uid,
                                                                                            pwd=pwd)


if __name__ == '__main__':
    print(database_interface('{SQL Server}', r'LEGION-Y9000P\SQLEXPRESS', 'SafeFace', 'sa', ''))
