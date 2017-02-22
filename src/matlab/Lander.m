classdef Lander
    properties
        s0; % Initial state
        st; % Target state
        c1; % Maximum thrust
        c2; % Effective 
        g;
        a;
        LBIndir;
        UBIndir;
    end
    methods
        function self = Lander(s0,st,c1,c2,g,a)
            self.s0 = s0;
            self.st = st;
            self.c1 = c1;
            self.c2 = c2;
            self.g  = g;
            self.a  = a;
            % Indirect method decision vector:
            % Z = [tf,lx,ly,lvx,lvy,lm]
            self.LBIndir = [0;-inf;-inf;-inf;-inf;-inf];
            self.UBIndir = [10000;inf;inf;inf;inf;inf];
        end
        function ds = EOM_State(self,s,c)
            x = s(1,1);
            y = s(2,1);
            vx = s(3,1);
            vy = s(4,1);
            m = s(5,1);
            u = c(1,1);
            st = c(2,1);
            ct = c(3,1);
            ds(1,1) = vx;
            ds(2,1) = vy;
            ds(3,1) = self.c1*u*st/m;
            ds(4,1) = self.c2*u*ct/m - self.g;
            ds(5,1) = -self.c1*u/self.c2;
        end
        function dl = EOM_Costate(self,fs,c)
            x = fs(1,1);
            y = fs(2,1);
            vx = fs(3,1);
            vy = fs(4,1);
            m = fs(5,1);
            lx = fs(6,1);
            ly = fs(7,1);
            lvx = fs(8,1);
            lvy = fs(9,1);
            lm = fs(10,1);
            u = c(1,1);
            st = c(2,1);
            ct = c(3,1);
            dl(1,1) = 0;
            dl(2,1) = 0;
            dl(3,1) = -lx;
            dl(4,1) = -ly;
            dl(5,1) = self.c1*u*(lvx*st+lvy*ct)/m^2;
        end
        function c = Pontryagin(self,fs)
            x = fs(1,1);
            y = fs(2,1);
            vx = fs(3,1);
            vy = fs(4,1);
            m = fs(5,1);
            lx = fs(6,1);
            ly = fs(7,1);
            lvx = fs(8,1);
            lvy = fs(9,1);
            lm = fs(10,1);
            lv = norm([lvx;lvy]);
            st = -lvx/lv;
            ct = -lvy/lv;
            if self.a == 1
                S = 1 - lm - lv*self.c2/m;
                if S >= 0
                    u = 0;
                elseif S < 0
                    u = 1;
                end
            else
                u = min(max((lm - self.a - lv*self.c2/m)/(2*self.c1*(1-self.a)),0),1);
            end
            c(1,1) = u;
            c(2,1) = st;
            c(3,1) = ct;
        end
        function dfs = EOM(self,t,fs)
            s = fs(1:5,1);
            c = self.Pontryagin(fs);
            ds = self.EOM_State(s,c);
            dl = self.EOM_Costate(fs,c);
            dfs = [ds;dl];
        end
        function [t,xf] = Shoot(self,decision)
            decision = reshape(decision, [6,1]);
            tf = decision(1,1);
            l0 = decision(2:6,1);
            fs0 = [self.s0;l0];
            [t,xf] = ode45(@self.EOM,[0,tf],fs0);
        end
        function H = Hamiltonian(self,fs)
            s = fs(1:5,1);
            l = fs(6:10,1);
            c = self.Pontryagin(fs);
            u = c(1,1);
            ds = self.EOM_State(s,c);
            H = 0;
            for n=1:5
                H = H + l(n,1)*ds(n,1);
            end
            H = H + self.a*self.c1/self.c2*u + (1-self.a)*self.c1^2/self.c2*u^2;
        end
        
            
        function no = Objective(self,decision)
            no = 1;
        end
        function [c,ceq] = Constraints(self,decision)
            [t,fs] = self.Shoot(decision);
            fs = fs(end,:)';
            ceq(1,1) = self.st(1,1) - fs(1,1);
            ceq(2,1) = self.st(2,1) - fs(2,1);
            ceq(3,1) = self.st(3,1) - fs(3,1);
            ceq(4,1) = self.st(4,1) - fs(4,1);
            ceq(5,1) = fs(10,1);
            ceq(6,1) = self.Hamiltonian(fs);
            c = [];
        end
        function dec = Optimise(self)
            func = @self.Objective;
            nvar = 6;
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            lb = self.LBIndir;
            ub = self.UBIndir;
            nonlcon = @self.Constraints;
            options = optimoptions(@ga,'UseParallel',true,...
                'Display','iter','HybridFcn',@fmincon,...
                'PlotFcn',@gaplotbestindiv);
            dec = ga(func,nvar,A,b,Aeq,beq,lb,ub,nonlcon,options);
        end
    end
end
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            