from collections.abc import Iterable
from numpy import   array, full_like, zeros_like, e,\
                    positive, negative, rint, absolute, fabs, sign, heaviside,\
                    conj, conjugate,\
                    add, subtract, multiply, matmul,\
                    divide, true_divide, floor_divide, remainder, mod, fmod, divmod,\
                    exp, exp2, power, float_power, log, log2, log10,\
                    sin, cos, tan

class Dual:
    """
    Implementation for automatic differentiation.
    """

    def __init__(self, real, dual):
        if(not isinstance(real, Iterable)):
            self.real = real
            self.dual = dual
            self.shape = ()
            self.dtype = type(real)
        elif(isinstance(dual, Iterable)):
            self.real = array(real)
            self.dual = array(dual, dtype=self.real.dtype)
            self.shape = self.real.shape
            self.dtype = self.real.dtype
        else:
            self.real = array(real)
            self.dual = full_like(real, dual, dtype=self.real.dtype)
            self.shape = self.real.shape
            self.dtype = self.real.dtype


    ##########################
    ## ARITHMETIC OPERATORS ##
    ##########################

    def __add__(self, other):
        if(isinstance(other, Dual)):
            real = self.real + other.real
            dual = self.dual + other.dual
            return Dual(real, dual)
        return Dual(self.real + other, self.dual)
    __radd__ = __add__

    def __sub__(self, other):
        if(isinstance(other, Dual)):
            real = self.real - other.real
            dual = self.dual - other.dual
            return Dual(real, dual)
        return Dual(self.real - other, self.dual)
    
    def __rsub__(self, other):
        if(isinstance(other, Dual)):
            real = other.real - self.real
            dual = other.dual - self.dual
            return Dual(real, dual)
        return Dual(other - self.real, self.dual)

    def __mul__(self, other):
        if(isinstance(other, Dual)):
            real = self.real * other.real
            dual = self.real * other.dual + self.dual * other.real
            return Dual(real, dual)
        return Dual(self.real * other, self.dual * other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if(isinstance(other, Dual)):
            real = self.real / other.real
            dual = (self.dual * other.real - self.real * other.dual) / (other.real * other.real)
            return Dual(real, dual)
        return Dual(self.real / other, self.dual / other)

    def __rtruediv__(self, other):
        if(isinstance(other, Dual)):
            real = other.real / self.real
            dual = (other.dual * self.real - other.real * self.dual) / (self.real * self.real)
            return Dual(real, dual)
        return Dual(other / self.real, - other * self.dual / (self.real * self.real))

    def __divmod__(self, other):
        return NotImplemented 
    __rdivmod__ = __floordiv__ = __rfloordiv__ = __divmod__

    def __pow__(self, other):
        if(isinstance(other, Dual)):
            return NotImplemented 
        real = pow(self.real, other)
        dual = other * pow(self.real, other - 1) * self.dual
        return Dual(real, dual)

    def __rpow__(self, other):
        if(isinstance(other, Dual)):
            return NotImplemented 
        real = pow(other, self.real)
        dual = real * self.dual * log(other)
        return Dual(real, dual)

    def __abs__(self):
        raise TypeError("abs not supported for instance of 'Dual'") 
        # return sqrt(self.real * self.real + self.dual * self.dual)

    #######################
    ## LOGICAL OPERATORS ##
    #######################

    def __bool__(self):
        return bool(self.real + self.dual)
    
    def __eq__(self, other):
        if(isinstance(other, Dual)):
            return self.real == other.real and self.dual == other.dual
        return self.real == other and self.dual == 0
    
    def __ne__(self, other):
        return not self == other
    
    def __gt__(self, other):
        return NotImplemented 
    __ge__ = __lt__ = __le__ = __gt__

    ####################
    ## REPRESENTATION ##
    ####################

    def __repr__(self):
        if(not self.shape):
            return f"{self.real}+{self.dual}eps"
        return f"array(reals={self.real},\n\tduals={self.dual}\n\tdtype={self.dtype}, shape={self.shape})"
    __str__ = __repr__

    #########################
    ## NUMPY COMPITABILITY ##
    #########################
    
    _sym_ufuncs = [add, subtract, negative, positive, rint, heaviside, sign]
    _mul_ufuncs = [multiply, matmul]
    _div_ufuncs = [divide, true_divide, floor_divide, remainder, mod, fmod]
    _pow_ufuncs = [exp, exp2, power, float_power]
    _log_ufuncs = [log, log2, log10]
    _rlp_ufuncs = [absolute, fabs]
    _cnj_ufuncs = [conj, conjugate]


    def __array__(self, dtype=None, copy=None):
        pass
        
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        n_inputs = len(inputs)
        reals = []
        duals = []
        for i in range(n_inputs):
            if(isinstance(inputs[i], Dual)):
                reals.append(inputs[i].real)
                duals.append(inputs[i].dual)
            else:
                reals.append(inputs[i])
                duals.append(zeros_like(inputs[i]) if isinstance(inputs[i], Iterable) else 0)

        real_result = ufunc(*reals, **kwargs)

        if   ufunc in Dual._sym_ufuncs:
            dual_result = ufunc(*duals, **kwargs)
        elif ufunc in Dual._mul_ufuncs:
            dual_result = ufunc(reals[0], duals[1], **kwargs) +\
                          ufunc(duals[0], reals[1], **kwargs)
        elif ufunc in Dual._div_ufuncs:
            dual_result = ufunc(-reals[0]*duals[1] + duals[0]*reals[1], 
                                reals[0]*reals[0], **kwargs)
        elif ufunc in Dual._pow_ufuncs:
            dual_result = ufunc(*reals, **kwargs) * duals[0] * log(ufunc(1 ,**kwargs))
        elif ufunc in Dual._log_ufuncs:
            dual_result = duals[0] / reals[0] * ufunc(e, **kwargs)
        elif ufunc in Dual._rlp_ufuncs:
            dual_result = 0
        elif ufunc in Dual._cnj_ufuncs:
            dual_result = -duals[0]
        elif ufunc == sin:
            dual_result = cos(reals[0], **kwargs) * duals[0]
        elif ufunc == cos:
            dual_result = -sin(reals[0], **kwargs) * duals[0]
        elif ufunc == tan:
            dual_result = duals[0] / cos(reals[0], **kwargs) / cos(reals[0], **kwargs)
        else:
            return NotImplemented

        return Dual(real_result, dual_result)


def diff(f, x, *args, **kwargs):
    return f(Dual(x, 1), *args, **kwargs).dual