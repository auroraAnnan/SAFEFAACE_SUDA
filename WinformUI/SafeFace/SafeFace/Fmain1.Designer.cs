namespace SafeFace
{
    partial class Fmain1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Fmain1));
            this.start = new Sunny.UI.UIButton();
            this.end = new Sunny.UI.UIButton();
            this.uiTextBox1 = new Sunny.UI.UITextBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.uiTextBox2 = new Sunny.UI.UITextBox();
            this.uiTextBox3 = new Sunny.UI.UITextBox();
            this.uiTextBox4 = new Sunny.UI.UITextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            // 
            // start
            // 
            this.start.Cursor = System.Windows.Forms.Cursors.Hand;
            this.start.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.start.Location = new System.Drawing.Point(500, 479);
            this.start.MinimumSize = new System.Drawing.Size(1, 1);
            this.start.Name = "start";
            this.start.Radius = 40;
            this.start.Size = new System.Drawing.Size(140, 50);
            this.start.TabIndex = 0;
            this.start.Text = "开始检测";
            this.start.TipsFont = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.start.Click += new System.EventHandler(this.uiButton1_Click);
            // 
            // end
            // 
            this.end.Cursor = System.Windows.Forms.Cursors.Hand;
            this.end.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.end.Location = new System.Drawing.Point(864, 479);
            this.end.MinimumSize = new System.Drawing.Size(1, 1);
            this.end.Name = "end";
            this.end.Radius = 40;
            this.end.Size = new System.Drawing.Size(140, 50);
            this.end.TabIndex = 1;
            this.end.Text = "停止检测";
            this.end.TipsFont = new System.Drawing.Font("微软雅黑", 12F);
            // 
            // uiTextBox1
            // 
            this.uiTextBox1.AutoScrollMargin = new System.Drawing.Size(1, 1);
            this.uiTextBox1.ButtonSymbolOffset = new System.Drawing.Point(0, 0);
            this.uiTextBox1.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.uiTextBox1.FillColorGradientDirection = System.Windows.Forms.FlowDirection.RightToLeft;
            this.uiTextBox1.FillReadOnlyColor = System.Drawing.Color.White;
            this.uiTextBox1.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.uiTextBox1.Location = new System.Drawing.Point(40, 79);
            this.uiTextBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.uiTextBox1.MaximumSize = new System.Drawing.Size(400, 600);
            this.uiTextBox1.MinimumSize = new System.Drawing.Size(1, 450);
            this.uiTextBox1.Multiline = true;
            this.uiTextBox1.Name = "uiTextBox1";
            this.uiTextBox1.Padding = new System.Windows.Forms.Padding(5);
            this.uiTextBox1.Radius = 10;
            this.uiTextBox1.ReadOnly = true;
            this.uiTextBox1.RectSides = System.Windows.Forms.ToolStripStatusLabelBorderSides.None;
            this.uiTextBox1.ScrollBarWidth = 1;
            this.uiTextBox1.ShowText = false;
            this.uiTextBox1.Size = new System.Drawing.Size(300, 450);
            this.uiTextBox1.Style = Sunny.UI.UIStyle.Custom;
            this.uiTextBox1.TabIndex = 2;
            this.uiTextBox1.TextAlignment = System.Drawing.ContentAlignment.TopLeft;
            this.uiTextBox1.Watermark = "";
            this.uiTextBox1.TextChanged += new System.EventHandler(this.uiTextBox1_TextChanged);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox1.Image")));
            this.pictureBox1.Location = new System.Drawing.Point(386, 132);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(360, 270);
            this.pictureBox1.TabIndex = 3;
            this.pictureBox1.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox2.Image")));
            this.pictureBox2.Location = new System.Drawing.Point(752, 132);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(360, 270);
            this.pictureBox2.TabIndex = 4;
            this.pictureBox2.TabStop = false;
            // 
            // uiTextBox2
            // 
            this.uiTextBox2.ButtonSymbolOffset = new System.Drawing.Point(0, 0);
            this.uiTextBox2.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.uiTextBox2.FillReadOnlyColor = System.Drawing.Color.FromArgb(((int)(((byte)(243)))), ((int)(((byte)(249)))), ((int)(((byte)(255)))));
            this.uiTextBox2.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.uiTextBox2.Location = new System.Drawing.Point(491, 92);
            this.uiTextBox2.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.uiTextBox2.MinimumSize = new System.Drawing.Size(1, 16);
            this.uiTextBox2.Name = "uiTextBox2";
            this.uiTextBox2.Padding = new System.Windows.Forms.Padding(5);
            this.uiTextBox2.ReadOnly = true;
            this.uiTextBox2.RectSides = System.Windows.Forms.ToolStripStatusLabelBorderSides.None;
            this.uiTextBox2.ShowText = false;
            this.uiTextBox2.Size = new System.Drawing.Size(149, 32);
            this.uiTextBox2.Style = Sunny.UI.UIStyle.Custom;
            this.uiTextBox2.TabIndex = 5;
            this.uiTextBox2.Text = "检测图像";
            this.uiTextBox2.TextAlignment = System.Drawing.ContentAlignment.MiddleCenter;
            this.uiTextBox2.Watermark = "";
            // 
            // uiTextBox3
            // 
            this.uiTextBox3.ButtonSymbolOffset = new System.Drawing.Point(0, 0);
            this.uiTextBox3.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.uiTextBox3.FillReadOnlyColor = System.Drawing.Color.FromArgb(((int)(((byte)(243)))), ((int)(((byte)(249)))), ((int)(((byte)(255)))));
            this.uiTextBox3.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.uiTextBox3.Location = new System.Drawing.Point(864, 92);
            this.uiTextBox3.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.uiTextBox3.MinimumSize = new System.Drawing.Size(1, 16);
            this.uiTextBox3.Name = "uiTextBox3";
            this.uiTextBox3.Padding = new System.Windows.Forms.Padding(5);
            this.uiTextBox3.ReadOnly = true;
            this.uiTextBox3.RectSides = System.Windows.Forms.ToolStripStatusLabelBorderSides.None;
            this.uiTextBox3.ShowText = false;
            this.uiTextBox3.Size = new System.Drawing.Size(149, 32);
            this.uiTextBox3.Style = Sunny.UI.UIStyle.Custom;
            this.uiTextBox3.TabIndex = 6;
            this.uiTextBox3.Text = "匹配图像";
            this.uiTextBox3.TextAlignment = System.Drawing.ContentAlignment.MiddleCenter;
            this.uiTextBox3.Watermark = "";
            // 
            // uiTextBox4
            // 
            this.uiTextBox4.ButtonSymbolOffset = new System.Drawing.Point(0, 0);
            this.uiTextBox4.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.uiTextBox4.FillReadOnlyColor = System.Drawing.Color.FromArgb(((int)(((byte)(243)))), ((int)(((byte)(249)))), ((int)(((byte)(255)))));
            this.uiTextBox4.Font = new System.Drawing.Font("微软雅黑", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.uiTextBox4.Location = new System.Drawing.Point(111, 37);
            this.uiTextBox4.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.uiTextBox4.MinimumSize = new System.Drawing.Size(1, 16);
            this.uiTextBox4.Name = "uiTextBox4";
            this.uiTextBox4.Padding = new System.Windows.Forms.Padding(5);
            this.uiTextBox4.ReadOnly = true;
            this.uiTextBox4.RectSides = System.Windows.Forms.ToolStripStatusLabelBorderSides.None;
            this.uiTextBox4.ShowText = false;
            this.uiTextBox4.Size = new System.Drawing.Size(149, 32);
            this.uiTextBox4.Style = Sunny.UI.UIStyle.Custom;
            this.uiTextBox4.TabIndex = 7;
            this.uiTextBox4.Text = "检测信息";
            this.uiTextBox4.TextAlignment = System.Drawing.ContentAlignment.MiddleCenter;
            this.uiTextBox4.Watermark = "";
            // 
            // Fmain1
            // 
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;
            this.ClientSize = new System.Drawing.Size(1145, 595);
            this.Controls.Add(this.uiTextBox4);
            this.Controls.Add(this.uiTextBox3);
            this.Controls.Add(this.uiTextBox2);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.uiTextBox1);
            this.Controls.Add(this.end);
            this.Controls.Add(this.start);
            this.ImeMode = System.Windows.Forms.ImeMode.Katakana;
            this.Name = "Fmain1";
            this.Style = Sunny.UI.UIStyle.Custom;
            this.StyleCustomMode = true;
            this.Text = "Form2";
            this.Initialize += new System.EventHandler(this.Fmain1_Initialize);
            this.Load += new System.EventHandler(this.FAside_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private Sunny.UI.UIButton start;
        private Sunny.UI.UIButton end;
        private Sunny.UI.UITextBox uiTextBox1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox2;
        private Sunny.UI.UITextBox uiTextBox2;
        private Sunny.UI.UITextBox uiTextBox3;
        private Sunny.UI.UITextBox uiTextBox4;
    }
}