from constants import Constants as const

# some physical constants
C3rd = 1.0/3.0
K_nr = 1.0036e13
K_er = 1.2435e15

def mu (X,Z):
    return 1/(0.75 + 1.25*X - 0.25*Z)

def mu_e (X):
    return 2/(1 + X)

def Tb_gas_rad (rho, X=0.7, Z=0.02):
#    mu = 1.0/(0.75 + 1.25*X - 0.25*Z)
#    mue = 2.0/(1 + X)
    T = ( 3*(const.R_gas/const.a_rad) * rho/mu(X,Z) )**C3rd
    return T

def Tb_gas_deg (rho, X=0.7, Z=0.02):
    n = 1.5
#    mu = 1.0/(0.75 + 1.25*X - 0.25*Z)
#    mue = 2.0/(1 + X)
    T_nr = (K_nr/const.R_gas) * mu(X,Z) * (rho**2/mu_e(X)**5)**C3rd
    T_er = (K_er/const.R_gas) * mu(X,Z) * (rho/mu_e(X)**4)**C3rd
    T = (1/T_nr**n + 1/T_er**n)**(-(1/n))
    # T = min(T_nr,T_er)
    return T


