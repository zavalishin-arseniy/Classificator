using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;
using Accord.Statistics.Testing;
using Accord;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Analysis;

namespace ProbaPera
{

    class Program
    {
        //то чего нет в Math
        public static double Factorial(int n)
        {
            int f = 1;
            for (int i = 1; i < n; i++)
                f *= i;
            return f;
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

        //распределения

        public static double ProbabilityNormalOne(double[] p, double[] value, double x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(-Pow(x - ev, 2) / (2 * ds * ds));
        }

        public static double ProbabilityNormal(double[] p, double[] value, double x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(-Pow(x - ev, 2) / (2 * ds * ds));
        }
        public static double RealityFunction(double[] p, double[] value, double[] x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            double sum = 0;
            for (int i = 1; i < x.Length; i++)
                sum += Sqrt(x[i] - ev);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(sum / (2 * ds * ds));
        }
        public static double Binary(double[] p, double[] value, double x)
        {
            double ds = Dispersion(p, value);
            double ev = ExpectedValue(p, value);
            return 1 / (Sqrt(2 * PI) * ds) * Exp(-Pow(x - ev, 2) / (2 * ds * ds));
        }
        public static double Puasson(int r, double l)
        {
            return Exp(-l) * Pow(l, r) / Factorial(r);
        }

        public static double ErrorNormal(double[] p, double[] v)
        {
            double er = 0;
            for (int i = 0; i < p.Length; i++)
                er += Pow((p[i] - ProbabilityNormal(p, v, p[i])), 2);
            return er;
        }

        // Logistic regression

        static double VectorProduct(double[] mul1, double[] mul2)
        {
            double product = 0.0;
            if (mul1.Length == mul2.Length)
            {
                for (int i = 0; i < mul1.Length; i++)
                {

                    product += mul1[i] * mul2[i];
                }
            }
            return product;
        }

        static double Normalise(double val)
        {
            return (1 / (1 + Math.Pow(Math.E, (-val))));
        }

        static double CalculateCost(double[] h, double[] outputs)
        {
            double cost = 0.0;
            if (h.Length == outputs.Length)
            {
                for (int i = 0; i < h.Length; i++)
                {
                        cost += (-outputs[i] * Math.Log(Math.Abs(h[i]))) - ((1 - outputs[i]) * Math.Log(1 - h[i]));
                }
            }
            return cost / h.Length;
        }

        static double[] CalculateTheta(double[][] inputs, double[] h, double[] outputs, double[] theta, double alpha)
        {
            double[] newtheta = new double[theta.Length];
            double[] delta = new double[theta.Length];
            for (int i = 0; i < theta.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    delta[i] += (h[j] - outputs[j]) * inputs[j][i];
                }
                delta[i] *= alpha;
            }
            for (int k = 0; k < delta.Length; k++)
            {
                newtheta[k] = theta[k] - delta[k];
            }

            return newtheta;
        }

        public static double[] Learn(double[][] inputs, double[] outputs, double[] theta, double alpha)
        {

            double[] h = new double[inputs.Length];
            int k = 0;
            double cost = 0.0;
            double[] newtheta = new double[theta.Length];
            while (true)
            {
                foreach (double[] input in inputs)
                {
                    h[k] = Normalise(VectorProduct(input, theta));
                    if (k + 1 < inputs.Length)
                    {
                        k++;
                    }
                }
                double newcost = CalculateCost(h, outputs);
                newtheta = CalculateTheta(inputs, h, outputs, theta, alpha);
                if (Sqrt(Math.Abs(Math.Abs(newcost) - Math.Abs(cost))) < 0.1)
                    break;
                else
                    cost = newcost;
                theta = newtheta;
            }
            foreach (var i in newtheta)
                Console.WriteLine(i);
            return h;
        }

        /*public static double LogistickRegression(double[] x, double[] y, int e)
        {
            double b0 = LogOdds(y);
            double[] b = new double[x.Length + 1];
            b[0] = b0;
            double[] z = new double[x.Length];
            double[] pn = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                z[i] = x[i] * b[i];
                pn[i] = Sigmoid(z[i]);
            }
        }*/
        [Obsolete]
        static void Main(string[] args)
        {
            /*int n = 9;
            double[] a = new double[n];
            double[,] c = new double[n,n];
            a[0] = 0.1;
            double sum = a[0];
            for (int i = 1; i < n; i++)
            {
                a[i] = a[i - 1] + 0.1;
                sum += a[i];
            }
            double[] b = new double[n];
            b[0] = 0.1;
            for (int i = 1; i < n / 2 + 1; i++)
                b[i] = b[i - 1] + 0.01;
            for (int i = n / 2 + 1; i < n; i++)
                b[i] = b[i - 1] - 0.01;

            for (int i = 1; i < n; i++)
            {
                c[1, i] = ProbabilityNormalOne(a, b, c[1, i]);
                c[0, i] = a[i];
            }





            // Console.WriteLine(Round(Dispersion(b, a)));
            //Console.WriteLine(Round(ExpectedValue(b, a)));
            Console.WriteLine(ErrorNormal(b, a));
            var chi = new ChiSquareTest(b, a, n-1);
            double pvalue = chi.PValue;
            bool significant = chi.Significant;

            Console.WriteLine("___________");
            for (int i = 0; i < 9; i++)
            {
                Console.Write(b[i]);
                Console.Write(' ');
                Console.WriteLine(Round(ProbabilityNormal(b, a, b[i]), 3));
            }

            Console.WriteLine(pvalue);
            Console.WriteLine(significant);
            Console.Read();*/
            double[][] inputs = new double[4][] { new double[] { 1, 2 }, new double[] { 1, 3 },
                new double[] { 1, 4 }, new double[] { 1, 5 } };

            double[] outputs = new double[] { 1, 1, 1, 0 };

            double alpha = 0.03;
            double[] theta = new double[] { 1, 1 };
            double[] newt = Learn(inputs, outputs, theta, alpha);
            Console.WriteLine("_____________");
            foreach (var i in newt) 
                Console.WriteLine(i);
            var lra = new LogisticRegressionAnalysis()
            {
                Regularization = 0.03
            };

            // compute the analysis
            LogisticRegression regression = lra.Learn(inputs, outputs);
            _ = Accord.Controls.DataGridBox.Show(regression.Coefficients);
            Console.ReadLine();
        }
    }
}
