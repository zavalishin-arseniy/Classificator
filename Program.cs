using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;


namespace ProbaPera
{

    class Program
    {
        //распределенеия
        public static double ProbabilityNormal(double[] p, double[] value, double x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(-Pow(x - ev, 2) / (2 * ds * ds));
        }
        public static double Binary(double[] p, double[] value, double x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(-Pow(x - ev, 2) / (2 * ds * ds));
        }
        public static double Sigmoid(double x)
        {
            return (1 / (1 + Exp(-x)));
        }
        public static double ExpectedValue(double[] p, double[] value)
        {
            double ev = 0;
            for (int i = 0; i < p.Length; i++)
                ev += p[i] * value[i];
            return ev;
        }
        public static double Dispersion(double[] p, double[] value)
        {
            double qs = 0;
            double ev = ExpectedValue(p, value);
            for (int i = 0; i < p.Length; i++)
                qs += (value[i] - ev) * (value[i] - ev);
            double ds = Sqrt(qs / value.Length);
            return ds;
        }
        public static double ProbabilitySucsess(double[] value)
        {
            int n = 0;
            for (int i = 0; i < value.Length; i++)
            {
                if (value[i] == 1)
                    n++;
            }
            return n / value.Length;
        }
        public static double ProbabilityUnsucsess(double[] value)
        {
            int n = 0;
            for (int i = 0; i < value.Length; i++)
            {
                if (value[i] == 0)
                    n++;
            }
            return n / value.Length;
        }
        public static double LogOdds(double[] value)
        {
            return Log(ProbabilitySucsess(value) / ProbabilityUnsucsess(value));
        }
        public static double LogistickRegression(double[] p, double[] value, int x)
        {
            double s = Sigmoid(-LogOdds(value));
            if (x == 1)
                return s;
            else
                return 1 / s;
        }
        static void Main(string[] args)
        {
        }
    }
}
