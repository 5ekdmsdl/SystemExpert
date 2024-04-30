#define CHECK_MPI(call) \
  do { \
    int code = call; \
    if (code != MPI_SUCCESS) { \
      char estr[MPI_MAX_ERROR_STRING]; \
      int elen; \
      MPI_Error_string(code, estr, &elen); \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr); \
      MPI_Abort(MPI_COMM_WORLD, code); \
    } \
  } while (0)
  