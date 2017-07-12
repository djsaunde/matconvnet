// @file vl_nnhebbian.cu
// @brief Hebbian learning layer MEX wrapper
// @author Daniel Saunders

/*
This file is a MEX wrapper for a custom Hebbian learning layer.
*/

#include "bits/mexutils.h"
#include "bits/nnhebbian.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>
#include <math.h>

/* option codes */
enum {
  opt_verbose = 0,
  opt_lambda,
  opt_eta,
  opt_connectivity,
  opt_cudnn,
  opt_no_cudnn
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,            opt_verbose           },
  {"Lambda",	       1,            opt_epsilon           },
  {"Eta",              1,            opt_moments           },
  {"Connectivity",     "8-lattice",  opt.connectivity      },
  {"Cudnn",            0,            opt_cudnn             },
  {"NoCudnn",          0,            opt_no_cudnn          },
  {0,                  0,            0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
   Resetting the context here resolves a crash when MATLAB quits and
   the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_WEIGHTS, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFMAPS, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  double lambda = 0.01 ;
  double eta = 0.001 ;
  char connectivity[] = '8-lattice' ;

  bool computeDerData = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 3) {
    mexErrMsgTxt("There are less than three arguments.") ;
  }
  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_lambda :
        if (!vlmxIsPlainScalar(optarg)) {
          mexErrMsgTxt("LAMBDA is not a plain scalar.") ;
        }
        lambda = mxGetPr(optarg)[0] ;
        break ;

      case opt_eta :
        if (!vlmxIsPlainScalar(optarg)) {
          exErrMsgTxt("ETA is not a plain scalar.") ;
        }
        eta = mxGetPr(optarg)[0] ;
        break ;

      case opt_connectivity :
        if (!vlmxIsString(optarg)) {
          exErrMsgTxt("CONNECTIVITY is not a string.") ;
        }
        connectivity = mxGetChars(optarg) ;
        break ;

      case opt_no_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
        break ;

      case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true) ;
#endif
        break ;
        
      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor weights(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  weights.init(in[IN_WEIGHTS]) ;
  weights.reshape(4) ;

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
  }

  /* Check for GPU/data class consistency */
  if (! vl::areCompatible(data, weights)) {
    mexErrMsgTxt("DATA and WEIGHTS do not have compatible formats.") ;
  }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }
  if (backMode && (data.getShape() != derOutput.getShape())) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have the same size.") ;
  }

  /* Get the filter geometry */
  vl::TensorShape weightsGeom(weights) ;
  if (weightsGeom.getHeight() != data.getDepth()) {
    mexErrMsgTxt("The WEIGHTS size does not match the DATA depth.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.init(deviceType, dataType, data.getShape()) ;
  } else {
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnbnorm: mode %s; %s; moments %s/%s\n",
              (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu",
              backMode?"backward":"forward",
              givenMomentsMode?"given":"computed",
              returnMomentsMode?"returned":"discared") ;
    vl::print("vl_nnbnorm: data: ", data) ;
    vl::print("vl_nnbnorm: multipliers: ", multipliers) ;
    vl::print("vl_nnbnorm: biases: ", biases) ;
    if (backMode) {
      vl::print("vl_nnbnorm: derOutput: ", derOutput) ;
      vl::print("vl_nnbnorm: derData: ", derData) ;
      vl::print("vl_nnbnorm: derMultipliers: ", derMultipliers) ;
      vl::print("vl_nnbnorm: derBiases: ", derBiases) ;
    } else {
      vl::print("vl_nnbnorm: output: ", output) ;
    }
    if (moments) { vl::print("vl_nnbnorm: moments: ", moments) ; }
    mexPrintf("vl_nnbnorm: epsilon: %f\n", epsilon) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  if (!backMode) {
    error = vl::nnhebbian_forward(context,
                                  output,
                                  moments,
                                  data,
                                  multipliers,
                                  biases) ;
    }
  } else {
    if (!givenMomentsMode) {
      error = vl::nnbnorm_backward(context,
                                   derData,
                                   derMultipliers,
                                   derBiases,
                                   moments,
                                   data,
                                   multipliers,
                                   biases,
                                   derOutput,
                                   epsilon);
    } else {
      error = vl::nnbnorm_backward_given_moments(context,
                                                 derData,
                                                 derMultipliers,
                                                 derBiases,
                                                 moments,
                                                 data,
                                                 multipliers,
                                                 biases,
                                                 derOutput,
                                                 epsilon) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (!backMode) {
    out[OUT_RESULT] = output.relinquish() ;
  } else {
    out[OUT_RESULT] = (computeDerData) ? derData.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERMULTIPLIERS] = (computeDerMultipliers)? derMultipliers.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERBIASES] = (computeDerBiases) ? derBiases.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
  }
  if (moments) {
    out[backMode ? 3 : 1] = moments.relinquish() ;
  }
}
