using ODE, JuMP, Ipopt

type PointLander2D
  s0::Vector{Float64}
  sf::Vector{Float64}
  c1::Float64
  c2::Float64
  g::Float64
  α::Float64

  Shoot::Function
  Optimise::Function

  function PointLander2D(s0,sf,c1,c2,g,α)
    self = new(s0,sf,c1,c2,g,α)

    function EOM_State(state::Vector, control::Vector)
      x,y,vx,vy,m = state
      u,sΘ,cΘ = control
      dx  = vx
      dy  = vy
      dvx = c1*u*sΘ/m
      dvy = c1*u*cΘ/m - g
      dm  = -c1*u/c2
      return [dx;dy;dvx;dvy;dm]
    end

    function EOM_Costate(fullstate::Vector, control::Vector)
      x,y,vx,vy,m,λx,λy,λvx,λvy,λm = fullstate
      u,sΘ,cΘ = control
      dλx  = 0
      dλy  = 0
      dλvx = -λx
      dλvy = -λy
      dλm  = c1*u*(λvx*sΘ+λvy*cΘ)/m^2
      return [dλx;dλy;dλvx;dλvy;dλm]
    end

    function Pontryagin(fullstate::Vector)
      x,y,vx,vy,m,λx,λy,λvx,λvy,λm = fullstate
      λv = √(λvx^2 + λvy^2)
      sΘ = -λvx/λv
      cΘ = -λvy/λv
      if α == 1
        S = 1 - λm - λv*c2/m
        if S ≥ 0
          u = 0
        elseif S < 0
          u = 1
        end
      else
        u = min(max((λm - α - λv*c2/m)/(2*c1*(1-α)),0),1)
      end
      return [u,sΘ,cΘ]
    end

    function EOM(t, fullstate::Vector)
      state   = fullstate[1:5]
      control = Pontryagin(fullstate)
      ds      = EOM_State(state, control)
      dλ      = EOM_Costate(fullstate, control)
      return vcat(ds,dλ)
    end

    function Shoot(decision::Vector)
      tf     = decision[1]
      λ0     = decision[2:6]
      fs0    = vcat(s0,λ0)
      t, fsl = ode78(EOM, fs0, [0;tf])
      x      = map(v -> v[1], fsl)
      y      = map(v -> v[2], fsl)
      vx     = map(v -> v[3], fsl)
      vy     = map(v -> v[4], fsl)
      λm     = map(v -> v[10], fsl)
      return [x[end]-sf[1],y[end]-sf[2],vx[end]-sf[3],vy[end]-sf[4],λm[end]]
    end

    function Hamiltonian(fullstate::Vector)
      s  = fullstate[1:5]
      λ  = fullstate[6:10]
      c  = Pontryagin(fullstate)
      ds = EOM_State(s, c)
      H  = 0
      for (λi, dsi) in zip(λ, s)
        H += λi*dsi
      end
      u,sΘ,cΘ = c
      H += ((1-α)*c1^2*u^2 + α*c1*u)/c2
      return H
    end

    self.Optimise = function()
      mod = Model(solver = IpoptSolver())
      @variable(mod, tf ≥ 0)
      @variable(mod, -1 ≤ λx0 ≤ 1)
      @variable(mod, -1 ≤ λy0 ≤ 1)
      @variable(mod, -1 ≤ λvx0 ≤ 1)
      @variable(mod, -1 ≤ λvy0 ≤ 1)
      @variable(mod, -1 ≤ λm0 ≤ 1)
      JuMP.register(:Shoot, 6, Shoot, autodiff=true)
      @NLconstraint(mod, Shoot())
      @NLexpression(mod, Shoot([tf,λx0,λy0,λvx0,λvy0,λm0]))
    end

    return self
  end
end

P = PointLander2D(
  [0;1000;0;0;10000], # Initial state
  [0;0;0;0;0],        # Final state
  44000,              # Max Thrust
  311*9.81,           # Effective velocity
  1.6229,             # Moon gravity
  1                   # Homotopy parameter
)

s  = P.s0
c  = [0.5,0.5,0.5]
λ  = [1.1,1.4,1.2,1.2,1.5]
fs = vcat(s,λ)
dec = [100.0;0.1;0.1;0.1;0.1;0.1]
P.Optimise()
