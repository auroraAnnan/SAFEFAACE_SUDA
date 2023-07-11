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
    public partial class Fmain3 : UIPage
    {
        public Fmain3()
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

        private void Fmain3_Load(object sender, EventArgs e)
        {

        }
    }
}
