/* The Computer Language Benchmarks Game
   https://salsa.debian.org/benchmarksgame-team/benchmarksgame/

   line-by-line from Greg Buchholz's C program
*/


class Program
{

    public static void Main(String[] args)
    {

        int w, h, x, y, bit_num = 0;
        int byte_acc = 0;
        int i, iter = 50;
        double limit = 2.0;
        double Zr, Zi, Cr, Ci, Tr, Ti;
        Stream stdout = Console.OpenStandardOutput();
        w = Int32.Parse(args[0]);
        h = w;

        Console.WriteLine("P4\n{0} {1}", w, h);

        for (y = 0; y < h; y++)
        {
            for (x = 0; x < w; x++)
            {
                Zr = 0.0; Zi = 0.0;
                Cr = (2 * (double)x / w - 1.5); Ci = (2 * (double)y / h - 1);

                for (i = 0; i < iter; i++)
                {
                    Tr = Zr * Zr - Zi * Zi + Cr;
                    Ti = 2 * Zr * Zi + Ci;
                    Zr = Tr; Zi = Ti;
                    if (Zr * Zr + Zi * Zi > limit * limit)
                        break;
                }

                if (Zr * Zr + Zi * Zi > limit * limit)
                    byte_acc = (byte_acc << 1) | 0x00;
                else
                    byte_acc = (byte_acc << 1) | 0x01;

                bit_num++;

                if (bit_num == 8)
                {
                    stdout.WriteByte((byte)byte_acc);
                    byte_acc = 0;
                    bit_num = 0;
                }
                else if (x == w - 1)
                {
                    byte_acc = byte_acc << (8 - w % 8);
                    stdout.WriteByte((byte)byte_acc);
                    byte_acc = 0;
                    bit_num = 0;
                }
            }
        }
    }

}