% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/main.R
\name{ManifoldContour}
\alias{ManifoldContour}
\title{Builds and plots  the mesh representing the desired isosurface}
\usage{
ManifoldContour(file_path, isovalue, verification, color_mesh, opacity,
  new_window)
}
\arguments{
\item{file_path}{A string: the path to the nhdr file}

\item{isovalue}{A number: the value corresponding to the desired isosurface}

\item{verification}{A boolean: determines whether the grid verification will be performed}

\item{color_mesh}{A string: the color to rendirind the resulting mesh}

\item{opacity}{A number: the opacity-level}

\item{new_window}{A boolean: determines whether a new view window will be opened}
}
\value{
The visualization of the generated mesh.
}
\description{
This function returns the visualization (rendered by the rgl package) of the mesh generated by the Marching Cubes 33 algotithm.
Optionally, the original dataset grid is preprocessed (subdivided in specifics points) to eliminate some configuration
which results in non-manifold edges.
}
\examples{
ManifoldContour(system.file("extdata", "f3.nhdr", package = "tcie"), 0.0, FALSE,"red",1.0, TRUE)
ManifoldContour(system.file("extdata", "f3.nhdr", package = "tcie"), 0.0, TRUE,"red",1.0, TRUE)
ManifoldContour(system.file("extdata", "f9.nhdr", package = "tcie"), 0.0, TRUE, "blue",1.0, TRUE)
}
