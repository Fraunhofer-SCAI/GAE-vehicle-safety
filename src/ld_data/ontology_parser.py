from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF

def load_ontology(file_path, ontology_ns):
    g = Graph()

    # # Define classes using the namespace
    # class_uri = ontology_ns.ClassName
    # g.add((class_uri, RDF.type, OWL.Class))


    # # Define properties using the namespace
    # property_uri = ontology_ns.hasProperty
    # g.add((property_uri, RDF.type, RDF.Property))


    # # Define individuals using the namespace
    # individual_uri = ontology_ns.IndividualName
    # g.add((individual_uri, RDF.type, OWL.NamedIndividual))

    g.parse(file_path, format="xml")


    return g


def extract_classes_and_properties(g, ontology_ns):
    classes_query = """
        SELECT DISTINCT ?class
        WHERE {
            ?class rdf:type owl:Class
        }
    """
    properties_query = """
        SELECT DISTINCT ?property
        WHERE {
            ?property rdf:type rdf:Property
        }
    """

    
    classes = [str(row[0]) for row in g.query(classes_query, initNs={'ontology_ns': ontology_ns})]
    properties = [str(row[0]) for row in g.query(properties_query, initNs={'ontology_ns': ontology_ns})]

    return classes, properties


def parse_ontology(file_path, base_uri):

    ontology_ns = Namespace(base_uri)
    ontology_graph = load_ontology(file_path, ontology_ns)
    # Your code to extract and process the necessary ontology data from the graph

    classes, properties = extract_classes_and_properties(ontology_graph, ontology_ns)

    # For this example, we will use a simple list of extracted data
    extracted_data = {'classes': classes, 'properties':properties}


    return extracted_data


