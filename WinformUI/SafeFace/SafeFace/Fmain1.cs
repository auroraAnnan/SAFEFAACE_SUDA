using Sunny.UI;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
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
            uiTextBox1.Text = "hello\r\nhjkhasjd\r\naisgdyag";
            Color c = Color.Blue;
            Color mc = Color.FromArgb(c.A, 243, 249, 255);
            uiTextBox2.FillReadOnlyColor = mc ;
        }

        private void FAside_Load(object sender, EventArgs e)
        {

        }

        private void Fmain1_Initialize(object sender, EventArgs e)
        {

        }

        private void uiButton1_Click(object sender, EventArgs e)
        {

        }

        private void uiTextBox1_TextChanged(object sender, EventArgs e)
        {
            
        }
    }
}
