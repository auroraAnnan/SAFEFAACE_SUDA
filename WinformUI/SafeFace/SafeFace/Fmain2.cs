using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Sunny.UI;

namespace SafeFace
{
    public partial class Fmain2 : UIPage
    {
        public Fmain2()
        {
            InitializeComponent();
            uiTextBox1.Text = "请点击开始检测以使用";
            Color c = Color.Blue;
            Color mc = Color.FromArgb(c.A, 243, 249, 255);
            uiTextBox4.FillReadOnlyColor = mc;
            uiTextBox2.FillReadOnlyColor = mc;
            uiTextBox3.FillReadOnlyColor = mc;
            uiTextBox1.FillReadOnlyColor = Color.White;
        }

        private void Fmain2_Load(object sender, EventArgs e)
        {

        }

        private void uiTextBox1_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
