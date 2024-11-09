from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class ModelParams:
    dense_units: int = 512
    freeze_backbone: bool = True
    dropout_rate: float = 0.5
    saved_model_name = "model.keras"


@dataclass
class TrainingParams:
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "binary_crossentropy"
    training_name: str = "training_name"


@dataclass
class Callback:
    type: str
    args: Dict[str, any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Callback to a dictionary format."""
        return asdict(self)


@dataclass
class Paths:
    subsets_path: str = ""
    labels_path: str = ""
    images_dir: str = ""


@dataclass
class BaldOrNotConfig:
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    training_params: TrainingParams = field(
        default_factory=lambda: TrainingParams()
    )
    callbacks: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            Callback(
                type="EarlyStopping",
                args={"monitor": "val_loss", "patience": 5},
            ).to_dict(),
            Callback(
                type="TensorBoard",
                args={"log_dir": "logs", "histogram_freq": 1},
            ).to_dict(),
        ]
    )
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    paths: Paths = field(default_factory=lambda: Paths())

    def __post_init__(self):
        self.model_params = (
            ModelParams(**self.model_params)
            if isinstance(self.model_params, dict)
            else self.model_params
        )
        self.training_params = (
            TrainingParams(**self.training_params)
            if isinstance(self.training_params, dict)
            else self.training_params
        )
        self.callbacks = [
            Callback(**params).to_dict()
            if isinstance(params, dict)
            else params
            for params in self.callbacks
        ]
        for callback in self.callbacks:
            if "args" not in callback:
                callback["args"] = {}
        self.paths = (
            Paths(**self.paths) if isinstance(self.paths, dict) else self.paths
        )


@dataclass
class GoogleApiData:
    """
    A dataclass to store authentication credentials for Google Custom Search API.

    Attributes:
        api_key (str): The API key for accessing Google APIs.
        cse_id (str): The Custom Search Engine ID (CSE ID) for identifying the custom search engine.

    Instructions to obtain `api_key` and `cse_id`:

    1. Getting the `api_key` (API Key) from Google Cloud Console:
       - Log into Google Cloud Console: https://console.cloud.google.com/
       - Create a new project or select an existing one:
           - Click the project selector at the top of the page and choose "New Project."
           - Enter a project name, then click "Create."
       - Enable the Google Custom Search API:
           - In the left-hand menu, go to "Library" and search for "Custom Search API."
           - Click "Enable" to activate the API for your project.
       - Generate an API Key:
           - Go to the "Credentials" section and click "Create Credentials."
           - Choose "API Key" from the dropdown. Your new API key will be generated and displayed.
           - Copy this key and assign it to `api_key` in this class.

       **Note**: Keep the API key secure. For enhanced security, restrict the key to specific APIs or IP addresses in Google Cloud Console.

    2. Getting the `cse_id` (Custom Search Engine ID):
       - Go to Google Custom Search Engine page: https://cse.google.com/cse/
       - Create a new custom search engine:
           - Click "New search engine."
           - Specify one or more sites for the search scope (e.g., `example.com`), or enter `*.com` to search across the web.
           - Click "Create" to finalize the custom search engine.
       - Retrieve the Custom Search Engine ID:
           - Open the settings of your new search engine.
           - Under the "Basics" section, find the "Search engine ID" — this is your `cse_id`.
           - Copy this ID and assign it to `cse_id` in this class.

    Security Reminder:
        Keep credentials private and avoid storing them in public repositories.
    """

    api_key: str = ""
    cse_id: str = ""


@dataclass
class SearchParams:
    query: str = "bald or not"
    num_images: int = 10


@dataclass
class DownloadParams:
    scrapper_output_path: str = r"C:\Users\user\Projekty\BaldOrNot\scrapping"


@dataclass
class FaceDetectionParams:
    min_face_area_ratio: float = 0.1
    scale_factor: float = 1.02
    min_neighbors: int = 5


@dataclass
class GoogleApiConfig:
    google_api_data: GoogleApiData = field(
        default_factory=lambda: GoogleApiData()
    )
    search_params: SearchParams = field(default_factory=lambda: SearchParams())
    download_params: DownloadParams = field(
        default_factory=lambda: DownloadParams()
    )
    face_detection_params: FaceDetectionParams = field(
        default_factory=lambda: FaceDetectionParams()
    )

    def __post_init__(self):
        self.google_api_data = (
            GoogleApiData(**self.google_api_data)
            if isinstance(self.google_api_data, dict)
            else self.google_api_data
        )
        self.search_params = (
            SearchParams(**self.search_params)
            if isinstance(self.search_params, dict)
            else self.search_params
        )
        self.download_params = (
            DownloadParams(**self.download_params)
            if isinstance(self.download_params, dict)
            else self.download_params
        )
        self.face_detection_params = (
            FaceDetectionParams(**self.face_detection_params)
            if isinstance(self.face_detection_params, dict)
            else self.face_detection_params
        )
