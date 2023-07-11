using Sunny.UI;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SafeFace
{
    public partial class Fmain1 : UIPage
    {
        public bool ifClick=false;
        public Fmain1()
        {
            InitializeComponent();
            uiTextBox1.Text = "请点击开始检测以使用";
            Color c = Color.Blue;
            Color mc = Color.FromArgb(c.A, 243, 249, 255);
            uiTextBox4.FillReadOnlyColor = mc ;
            uiTextBox2.FillReadOnlyColor = mc;
            uiTextBox3.FillReadOnlyColor = mc;
            uiTextBox1.FillReadOnlyColor = Color.White;

            //-------------------
            
        }
        static byte[] ReadData(NetworkStream stream)
        {
            byte[] buffer = new byte[1024];
            using (MemoryStream ms = new MemoryStream())
            {
                int bytesRead;
                while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    ms.Write(buffer, 0, bytesRead);
                }
                return ms.ToArray();
            }
        }

        static Image ByteArrayToImage(byte[] byteArray)
        {
            using (MemoryStream ms = new MemoryStream(byteArray))
            {
                return Bitmap.FromStream(ms, true);
          
            }
        }

        static Bitmap ConvertToFormat(Bitmap bitmap, PixelFormat format)
        {
            Bitmap newBitmap = new Bitmap(bitmap.Width, bitmap.Height, format);
            using (Graphics g = Graphics.FromImage(newBitmap))
            {
                g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
            }
            return newBitmap;
        }

        private void FAside_Load(object sender, EventArgs e)
        {

        }

        private void Fmain1_Initialize(object sender, EventArgs e)
        {

        }

        private void uiButton1_Click(object sender, EventArgs e)
        {
            int port = 8888;  // 与Python客户端相同的端口号
            TcpListener listener = new TcpListener(IPAddress.Any, port);
            listener.Start();

            Console.WriteLine("等待Python客户端连接...");
            TcpClient client = listener.AcceptTcpClient();
            Console.WriteLine("已连接到Python客户端.");

            // 读取图像字节流
            byte[] imageBytes = ReadData(client.GetStream());
            Console.WriteLine(imageBytes[0]);
            Console.WriteLine(imageBytes[1]);
            Console.WriteLine(imageBytes[2]);
            Console.WriteLine(imageBytes[3]);
            // 将字节流转换为图像
            Image image = ByteArrayToImage(imageBytes);

            // 在此处进行图像处理操作
            // ...
            pictureBox1.Image = image;
            // 关闭连接
            client.Close();
            listener.Stop();
        }

        private void uiTextBox1_TextChanged(object sender, EventArgs e)
        {
            
        }
    }
}
