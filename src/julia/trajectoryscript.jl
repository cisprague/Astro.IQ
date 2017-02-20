# x = [tf,y1,u1,y2b,u2b,y2,u2,...,ymb,umb,ym,um]
using JuMP, Ipopt
prob = Model(solver=IpoptSolver(print_level=1))
M = 200

# Parametres
@NLparameter(prob, I_sp == 311)
@NLparameter(prob, g_0 == 9.812)
@NLparameter(prob, g_l == 1.6229)

# Time
@variable(prob, 1 <= t_f <= 1000)
@NLexpression(prob, τ, 1/(M-1))


# States
@variable(prob, -1000 <= x[1:M] <= 1000)
@variable(prob, 0 <= y[1:M] <= 1000)
@variable(prob, -400 <= vx[1:M] <= 400)
@variable(prob, -400 <= vy[1:M] <= 400)
@variable(prob, 0 <= m[1:M] <= 10000)

# Controls
@variable(prob, 0 <= T[1:M] <= 44000)
@variable(prob, 0 <= ϕ[1:M] <= π)

# Hermite-Simpson-Sperated Variables
@variable(prob, -1000 <= xb[2:M] <= 1000)
@variable(prob, 0 <= yb[2:M] <= 1000)
@variable(prob, -100 <= vxb[2:M] <= 100)
@variable(prob, -100 <= vyb[2:M] <= 100)
@variable(prob, 0 <= mb[2:M] <= 10000)
@variable(prob, 0 <= Tb[2:M] <= 44000)
@variable(prob, 0 <= ϕb[2:M] <= π)

# Objective
@objective(prob, Max, m[M])

# Constraints
@constraint(prob, x[1] == 0)
@constraint(prob, y[1] == 1000)
@constraint(prob, vx[1] == -5)
@constraint(prob, vy[1] == -3)
@constraint(prob, m[1] == 10000)

@constraint(prob, x[M] == 0)
@constraint(prob, y[M] == 0)
@constraint(prob, vx[M] == 0)
@constraint(prob, vy[M] == 0)


@NLexpression(prob, ax[j=1:M], (T[j]*sin(ϕ[j])/m[j]))
@NLexpression(prob, ay[j=1:M], (T[j]*cos(ϕ[j])/m[j]) - g_l)
@NLexpression(prob, md[j=1:M], -T[j]/(I_sp*g_0))
@NLexpression(prob, axb[j=2:M], (Tb[j]*sin(ϕb[j])/mb[j]))
@NLexpression(prob, ayb[j=2:M], (Tb[j]*cos(ϕb[j])/mb[j]) - g_l)
@NLexpression(prob, mdb[j=2:M], -Tb[j]/(I_sp*g_0))

# Hermite-Simpson-Seperated Constraints
for k in 1:M-1
    # Hermite Interplotation
    @NLconstraint(prob,
    xb[k+1] == 0.5*(x[k+1] + x[k]) + ((τ*t_f)/(8))*(vx[k] - vx[k+1])
    )
    @NLconstraint(prob,
    yb[k+1] == 0.5*(y[k+1] + y[k]) + ((τ*t_f)/(8))*(vy[k] - vy[k+1])
    )
    @NLconstraint(prob,
    vxb[k+1] == 0.5*(vx[k+1] + vx[k]) + ((τ*t_f)/(8))*(ax[k] - ax[k+1])
    )
    @NLconstraint(prob,
    vyb[k+1] == 0.5*(vy[k+1] + vy[k]) + ((τ*t_f)/(8))*(ay[k] - ay[k+1])
    )
    @NLconstraint(prob,
    mb[k+1] == 0.5*(m[k+1] + m[k]) + ((τ*t_f)/(8))*(md[k] - md[k+1])
    )

    # Simpson Quadrature
    @NLconstraint(prob,
    x[k+1] == x[k] + ((τ*t_f)/6)*(vx[k+1] + 4*vxb[k+1] + vx[k])
    )
    @NLconstraint(prob,
    y[k+1] == y[k] + ((τ*t_f)/6)*(vy[k+1] + 4*vyb[k+1] + vy[k])
    )
    @NLconstraint(prob,
    vx[k+1] == vx[k] + ((τ*t_f)/6)*(ax[k+1] + 4*axb[k+1] + ax[k])
    )
    @NLconstraint(prob,
    vy[k+1] == vy[k] + ((τ*t_f)/6)*(ay[k+1] + 4*ayb[k+1] + ay[k])
    )
    @NLconstraint(prob,
    m[k+1] == m[k] + ((τ*t_f)/6)*(md[k+1] + 4*mdb[k+1] + md[k])
    )
end

solve(prob)

using PyPlot

plot(x=getValue(x),y=getValue(y))
