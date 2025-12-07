def classFactory(iface):
    from .sql_query_generator import SQLQueryGeneratorPlugin
    return SQLQueryGeneratorPlugin(iface)
    
