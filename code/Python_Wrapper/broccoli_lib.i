%module broccoli
%{
  /* Includes the header in the wrapper code */
  #include "../BROCCOLI_LIB/broccoli_lib.h"
%}
 
/* Parse the header file to generate wrappers */

%include exception.i

%include carrays.i
%array_class(float, floatArray);
%array_class(cl_float2, cl_float2Array);


%typemap(out) int *
{
    $result = PyList_New(200);
    for (int i = 0; i < 200; ++i) {
        PyList_SetItem($result, i, PyInt_FromLong($1[i]));
    }
}

%ignore Coords3D::operator[];
%include "../BROCCOLI_LIB/broccoli_constants.h"
%include "../BROCCOLI_LIB/broccoli_lib.h"

typedef unsigned int cl_uint;

%extend Coords3D
{
    int __getitem__(int i) const
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return 0;
        }
        return (*$self)[i];
    }
    
    void __setitem__(int i, int v)
    {
        if (i < 0 || i > 2)
        {
            SWIG_Error(SWIG_IndexError, "index must be between 0 and 2");
            return;
        }
        (*$self)[i] = v;
    }
    
    const char* __repr__() const
    {
        char* buf = (char*)malloc(64);
        sprintf(buf, "Coords3D(%d, %d, %d)", (*$self)[0], (*$self)[1], (*$self)[2]);
        return buf;
    }
}
