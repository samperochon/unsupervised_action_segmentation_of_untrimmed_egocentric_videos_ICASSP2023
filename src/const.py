DEFAULT_DEVICE = 'cuda'

DEFAULT_TAU_SAMPLE = 1

MIN_SEGMENTS_SIZE = .5 # .5s is the minimum possible segment size 
DEFAULT_PENALTY = 20

DEFAULT_MODEL_NAME = 'resnet152'

DEFAULT_REMOVE_OUTLIERS = True
DEFAULT_OUTLIER_METHODS = ['custom', 'silhouette_score']
DEFAULT_OUTLIERS_VOTING_SCHEME = 'union'

DEFAULT_REDUCE_EMBEDDING=True
DEFAULT_EMBEDDING_DIMENSION_REDUCTION=100

VERBOSITY = 0 # Debug mode

DEFAULT_CONFIG = {# VideoFrameDataset
                  'video_path': None,
                  # Pipeline
                  'tau_sample':DEFAULT_TAU_SAMPLE,
                  'model_name' : DEFAULT_MODEL_NAME,

                  # Outliers removal
                  'remove_outliers' : DEFAULT_REMOVE_OUTLIERS,
                  'outlier_methods': DEFAULT_OUTLIER_METHODS,
                  'voting_scheme': DEFAULT_OUTLIERS_VOTING_SCHEME,

                  # Embedding
                  'reduce_embedding': DEFAULT_REDUCE_EMBEDDING,
                  'dim_embedding': DEFAULT_EMBEDDING_DIMENSION_REDUCTION,
                  'verbosity': VERBOSITY}