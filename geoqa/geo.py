# adapted from Jacob Andreas's data readers for Neural Module Networks, https://github.com/jacobandreas/nmn2

from geoqa.utils import parse_tree

from collections import namedtuple
import numpy as np

DATA_FILE = "data/geo/environments/%s/training.txt"
PARSE_FILE = "data/geo/environments/%s/training.sps"
WORLD_FILE = "data/geo/environments/%s/world.txt"
LOCATION_FILE = "data/geo/environments/%s/locations.txt"

STATES = ["fl", "ga", "mi", "nc", "ok", "pa", "sc", "tn", "va", "wv"]

MAX_LITERALS = 7

# note: these are not standard folds for the dataset
DEV_ENVS_BY_FOLD = {
    1: ["va", "wv"],
    2: ["fl", "ga"],
    3: ["mi", "nc"],
    4: ["ok", "pa"],
    5: ["sc", "tn"],
}

TEST_ENVS_BY_FOLD = {
    1: ["fl", "ga"],
    2: ["mi", "nc"],
    3: ["ok", "pa"],
    4: ["sc", "tn"],
    5: ["va", "wv"],
}

# LFS
LAMBDA = 'lambda'
EXISTS = 'exists'
AND = 'and'

VARS = ['$w', '$x', '$y', '$z']

CATS = ["city", "state", "park", "island", "beach", "ocean", "lake", "forest",
        "major", "peninsula", "capital", "body", "water", "salt", "fresh", "place"]
RELS = ["in-rel", "on-rel",
        "north-rel", "south-rel", "east-rel", "west-rel",
        "northeast-rel", "southeast-rel", "southwest-rel", "northwest-rel",
        "border-rel", "capital-rel",
        "near-rel", "close-rel",
        "surround-rel", "bigger-rel", "abut-rel", "along-rel", "inside-rel", "contain-rel"
        ]
_World = namedtuple("_World", ["name", "entities", "categories", "relations", "entity_predicates", "index_to_entities"])

class World(_World):
    def __repr__(self):
        return "<World: {} with {} entities, {} categories, {} relations>".format(
            self.name, len(self.entities), (self.categories > 0).sum(), (self.relations > 0).sum()
        )

    def print(self):
        print("name: {}".format(self.name))
        print("entities:")
        for entity in self.index_to_entities:
            print("\t{}".format(entity))
        print("categories:")
        for cat_ix, cat in enumerate(CATS):
            for ent_ix, on in enumerate(self.categories[cat_ix]):
                if on[0] == 1:
                    print("\t{}({})".format(cat, self.index_to_entities[ent_ix]))
        print("relations:")
        for rel_ix, rel in enumerate(RELS):
            for ent_ix, ons in enumerate(self.relations[rel_ix]):
                for ent_jx, on in enumerate(ons):
                    if on == 1:
                        print("\t{}({}, {})".format(
                            rel, 
                            self.index_to_entities[ent_ix],
                            self.index_to_entities[ent_jx],
                        ))

    def get_denotation_from_answer(self, answer):
        if answer in ['no', 'yes']:
            answer_denotation = answer
        else:
            answer_entities = answer.split(',') if answer.strip() else []
            answer_denotation = set([ans for ans in answer_entities])
        return answer_denotation

CAT_KEYWORDS = {
    cat: [cat] for cat in CATS
}

REL_KEYWORDS = {
    rel: [rel.split('-')[0]]
    for rel in RELS
}

REL_KEYWORDS['surround-rel'].extend(['surrounds', 'surrounded'])
REL_KEYWORDS['abut-rel'].extend(['abuts'])
REL_KEYWORDS['contain-rel'].extend(['contains'])
REL_KEYWORDS['border-rel'].extend(['bordering', 'borders'])
REL_KEYWORDS['in-rel'].extend(['near'])

CAT_KEYWORDS['state'].extend(['states'])
CAT_KEYWORDS['body'].extend(['bodies'])
CAT_KEYWORDS['city'].extend(['cities'])
CAT_KEYWORDS['lake'].extend(['lakes'])
CAT_KEYWORDS['beach'].extend(['beaches'])
CAT_KEYWORDS['forest'].extend(['forests'])
CAT_KEYWORDS['ocean'].extend(['oceans'])
CAT_KEYWORDS['park'].extend(['parks'])
CAT_KEYWORDS['place'].extend(['places'])

DATABASE_SIZE=10

YES = "yes"
NO = "no"

# map from entities to words that compose it
ENTITY_KEYWORDS = {}

ENTITY_KEYWORDS_BY_STATE = {}

ENTITY_ACTIONS_TO_ENTITIES = {}

def read_world(state_name):
    places = list()
    with open(LOCATION_FILE % state_name) as loc_f:
        for line in loc_f:
            parts = line.strip().split(";")
            places.append(parts[0])

    cats = {place: np.zeros((len(CATS),)) for place in places}
    rels = {(pl1, pl2): np.zeros((len(RELS),)) for pl1 in places for pl2 in places}

    with open(WORLD_FILE % state_name) as world_f:
        for line in world_f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            name = parts[0][1:]
            places_here = parts[1].split(",")
            if name in CATS:
                cat_id = CATS.index(name)
                for place in places_here:
                    cats[place][cat_id] = 1
            elif name in RELS:
                rel_id = RELS.index(name)
                for place_pair in places_here:
                    if not place_pair.strip():
                        continue
                    pl1, pl2 = place_pair.split("#")
                    rels[pl1, pl2][rel_id] = 1
                    if rels[pl2,pl1][rel_id] != 1:
                        rels[pl2, pl1][rel_id] = -1


    clean_places = [p.lower().replace(" ", "_") for p in places]
    entity_predicates = []
    for pl in clean_places:
        entity_predicate = 'kb-'+pl
        entity_predicates.append(entity_predicate)
        keywords = pl.split('_')
        ENTITY_KEYWORDS[entity_predicate] = keywords
        ENTITY_ACTIONS_TO_ENTITIES[entity_predicate] = pl
        if state_name not in ENTITY_KEYWORDS_BY_STATE:
            ENTITY_KEYWORDS_BY_STATE[state_name] = {}
        ENTITY_KEYWORDS_BY_STATE[state_name][entity_predicate] = keywords
    place_index = {place: i for (i, place) in enumerate(places)}
    clean_place_index = {place: i for (i, place) in enumerate(clean_places)}
    categories = np.zeros((len(CATS), DATABASE_SIZE, 1))
    relations = np.zeros((len(RELS), DATABASE_SIZE, DATABASE_SIZE))

    for p1, i_p1 in place_index.items():
        categories[:, i_p1, 0] = cats[p1]
        for p2, i_p2 in place_index.items():
            relations[:, i_p1, i_p2] = rels[p1, p2]

    world = World(
        name=state_name,
        entities=clean_place_index,
        categories=categories,
        relations=relations,
        entity_predicates=entity_predicates,
        index_to_entities=clean_places,
    )

    return world

# populate ENTITY_KEYWORDS and ENTITY_ACTIONS_TO_ENTITIES
for state_name in STATES:
    read_world(state_name)

def remap(mapping, sexp):
    if isinstance(sexp, tuple):
        sexp = tuple(remap(mapping, sub_sexp) for sub_sexp in sexp)
        if sexp[0] == EXISTS:
            variables = tuple(sorted(sexp[1:-1]))
            sexp = (EXISTS,) + variables + (sexp[-1],)
        return sexp
    else:
        return mapping.get(sexp, sexp)

def reorder_variables(sexp, only_reorder_in_and):
    var_ordering = {}
    def reorder_variables_helper(sexp, in_and):
        if sexp[0] == AND:
            in_and = True
        if isinstance(sexp, tuple):
            for sub_sexp in sexp:
                reorder_variables_helper(sub_sexp, in_and)
        elif sexp.startswith('$') and ((not only_reorder_in_and) or in_and):
            var = sexp
            if var not in var_ordering:
                var_ordering[var] = len(var_ordering)

    reorder_variables_helper(sexp, False)

    mapping = {
        var: VARS[order]
        for var, order in var_ordering.items()
    }
    return remap(mapping, sexp)

def alphabetize_clauses(sexp):
    if isinstance(sexp, tuple):
        if sexp[0] == AND:
            sorted_sexp = tuple(sorted(sexp[1:]))
            return (AND,) + sorted_sexp
        else:
            return tuple(alphabetize_clauses(sub_sexp) for sub_sexp in sexp)
    else:
        return sexp

def read_data(state_name):
    questions, answers = [], []
    lambdas = []
    reorder_sexp_variables = True
    alphabetize_sexp_clauses = False
    only_reorder_in_and = True
    with open(DATA_FILE % state_name) as data_f:
        for line in data_f:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue

            parts = line.split(";")

            question = parts[0]
            if question[-1] != "?":
                question += " ?"
            question = question.lower()
            questions.append(question)

            answer = parts[1].lower().replace(" ", "_")
            answers.append(answer)

            if len(parts) >= 4:
                lambd = parts[3]
                if lambd.strip():
                    sexp = parse_tree(lambd.lower())
                    if only_reorder_in_and:
                        if alphabetize_sexp_clauses:
                            sexp = alphabetize_clauses(sexp)
                        if reorder_sexp_variables:
                            sexp = reorder_variables(sexp, True)
                    else:
                        # order swapped for backward compatibility
                        if reorder_sexp_variables:
                            sexp = reorder_variables(sexp, False)
                        if alphabetize_sexp_clauses:
                            sexp = alphabetize_clauses(sexp)
                    lambdas.append(sexp)
                else:
                    lambdas.append(None)
            else:
                lambdas.append(None)

    assert len(questions) == len(answers) == len(lambdas)

    return questions, answers, lambdas

