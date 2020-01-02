import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def create_pca_embedding(data: pd.DataFrame, n_components: int = 50, create_plots=True) -> pd.DataFrame:
    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=n_components)

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=data.index,
        columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
    )
    # Create explained variance ratio plot
    if create_plots:
        plt.figure()
        plt.plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )
        # Plot the cumulative explained variance
        plt.plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )
        plt.legend()
        plt.title("Explained variance per PCA component")
        plt.show()

    return pca_df
