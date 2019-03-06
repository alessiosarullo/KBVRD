from typing import Dict, Type, Set

from lib.models.abstract_model import AbstractHOIModel


def get_all_models_by_name() -> Dict[str, Type[AbstractHOIModel]]:
    # This is needed because otherwise subclasses are not registered. FIXME maybe?
    from lib.models.base_model import BaseModel
    from lib.models.nmotifs.hoi_nmotifs import HOINMotifs, HOINMotifsHybrid

    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

    all_model_classes = get_all_subclasses(AbstractHOIModel)  # type: Set[Type[AbstractHOIModel]]
    all_model_classes_dict = {}
    for model in all_model_classes:
        try:
            all_model_classes_dict[model.get_cline_name()] = model
        except NotImplementedError:
            pass
    print(all_model_classes_dict)
    return all_model_classes_dict
