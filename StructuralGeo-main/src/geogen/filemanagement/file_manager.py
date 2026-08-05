import os
import pickle as pickle

from geogen.model import GeoModel


class FileManager:
    """
    Handles file operations for GeoModel instances including saving and loading models
    from disk, applying an operation to all saved instances in a folder.
    Supports both automatic indexing and manual naming for files.

    Parameters
    ----------
    base_dir : str, optional
        The directory where models are saved and loaded. Default is '../saved_models'.
    auto_index : bool, optional
        Automatically manage file naming and indexing if True. Default is True.
    """

    def __init__(self, base_dir="../saved_models", auto_index=True):
        self.base_dir = base_dir
        self.file_index = None
        self.auto_index = auto_index

    def _get_initial_file_index(self, sequential=True):
        """Determine the starting file index based on existing files in the directory."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            return 0
        existing_files = [f for f in os.listdir(self.base_dir) if f.endswith(".pkl")]
        if not existing_files:
            return 0
        # Extract indexes from file names assuming the format 'model_<index>.pkl'
        indexes = [int(f.split("_")[-1].split(".")[0]) for f in existing_files]
        return max(indexes) + 1 if indexes else 0

    def _file_save_string(self, model_index):
        """Format for saving model files."""

        return f"model_{model_index}.pkl"

    def save_geo_model(self, geo_model, save_dir, filename=None, lean=True):
        """
        Saves a GeoModel instance to a specified directory.

        Parameters
        ----------
        geo_model : GeoModel
            The GeoModel instance to be saved.
        lean : bool
            Determines the mode of saving the model. If True, the model is saved without the 'data' attribute,
            including only essential serialized parameters such as history and bounds.

        Notes
        -----
        This function allows for a 'lean' save, where non-essential data is excluded from the saved file to reduce
        file size and optimize storage. This is particularly useful in environments where storage efficiency is critical.

        Examples
        --------
        To save a GeoModel instance without non-essential data:

        >>> my_geo_model = GeoModel()
        >>> save_geo_model(my_geo_model, lean=True)
        """
        if lean:
            # Implement clear_data if needed to remove unnecessary large data
            geo_model.clear_data()

        if self.auto_index:
            self.file_index = self._get_initial_file_index()
            file_path = os.path.join(save_dir, self._file_save_string(self.file_index))
            self.file_index += 1
        else:
            if filename is None:
                raise ValueError("Filename must be provided if auto_index saving is disabled.")
            else:
                file_path = os.path.join(save_dir, filename + ".pkl")

        with open(file_path, "wb") as file:
            pickle.dump(geo_model, file)
        print(f"Model saved to {file_path}")

    def load_geo_model(self, file_path):
        """
        Load a GeoModel instance from a file.

        Parameters
        ----------
        file_path : str
            The path to the file from which the GeoModel will be loaded.

        Returns
        -------
        GeoModel
            The loaded GeoModel instance.
        """
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model

    def save_all_models(self, models, lean=True):
        """
        Saves a list of GeoModel instances.

        Parameters
        ----------
        models : list of GeoModel
            The GeoModel instances to save.
        lean : bool, optional
            If True, the models will be saved without the 'data' attribute. Default is True.
        """
        for model in models:
            self.save_geo_model(model, self.base_dir, lean=lean)

    def walk_and_process_models(self, action_callback, *args, **kwargs):
        """
        Walk through directories and apply a callback to each model file.

        Parameters
        ----------
        action_callback : callable
            The callback function to apply to each model file.
        """
        print(f"Processing models in {self.base_dir}")
        for root, dirs, files in os.walk(self.base_dir):
            file_list = [os.path.join(root, file) for file in files if file.endswith(".pkl")]
            if self.auto_index:
                file_list.sort(
                    key=lambda x: (
                        os.path.dirname(x),
                        int(os.path.basename(x).split("_")[-1].split(".")[0]),
                    )
                )
            for file_path in file_list:
                action_callback(file_path, *args, **kwargs)

    def load_all_models(self):
        """
        Loads all GeoModel instances from the file system.

        Returns
        -------
        list of GeoModel
            A list of loaded GeoModel instances.
        """
        models = []
        self.walk_and_process_models(lambda file_path: models.append(self.load_geo_model(file_path)))
        return models

    """ Model renewal functions for updating or pickled models with changes in the model class."""

    def renew_all_models(self, new_save_dir):
        """Process each model and save it to a new directory while preserving the directory structure.

        Used for updating models with new features or changes in the model class."""
        for root, dirs, files in os.walk(self.base_dir):
            file_list = [os.path.join(root, file) for file in files if file.endswith(".pkl")]
            file_list.sort(
                key=lambda x: (
                    os.path.dirname(x),
                    int(os.path.basename(x).split("_")[-1].split(".")[0]),
                )
            )
            for file_path in file_list:
                model: GeoModel = self.load_geo_model(file_path)
                model._validate_model_params()
                # Make any alterations needed here
                self.update_model_version(model)
                folder_name = os.path.basename(os.path.dirname(file_path))
                model.name = folder_name
                # Create a new file path by replacing the base directory with the save directory
                new_file_path = os.path.join(new_save_dir, os.path.relpath(file_path, self.base_dir))

                # Ensure the directory exists
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

                # Save the model to the new location
                with open(new_file_path, "wb") as file:
                    pickle.dump(model, file)
                    print(f"Model saved to {new_file_path}")

    def update_model_version(self, model):
        """Update the model version to the latest version."""
        pass


if __name__ == "__main__":
    file_manager = FileManager(base_dir="./database")
    file_manager.renew_all_models("./new_database")
