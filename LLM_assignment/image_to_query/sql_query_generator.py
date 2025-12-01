from qgis.PyQt.QtWidgets import QAction, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QSettings
from qgis.core import QgsVectorLayer, QgsProject, QgsDataSourceUri
from qgis.utils import iface
from PIL import Image
import google.generativeai as genai

class SQLQueryGeneratorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SQL Query Generator")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        image_layout = QHBoxLayout()
        self.image_input = QLineEdit()
        self.image_input.setPlaceholderText("Select an image...")
        image_btn = QPushButton("Browse Image")
        image_btn.clicked.connect(self.browse_image)
        image_layout.addWidget(QLabel("Image:"))
        image_layout.addWidget(self.image_input)
        image_layout.addWidget(image_btn)
        layout.addLayout(image_layout)
        
        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.setMaximumHeight(80)
        prompt_layout.addWidget(self.prompt_input)
        layout.addLayout(prompt_layout)
        
        generate_btn = QPushButton("Generate SQL Query")
        generate_btn.clicked.connect(self.generate_query)
        layout.addWidget(generate_btn)
        
        layout.addWidget(QLabel("Generated SQL Query:"))
        self.sql_output = QTextEdit()
        self.sql_output.setReadOnly(True)
        layout.addWidget(self.sql_output)
        
        edit_btn = QPushButton("Edit Query")
        edit_btn.clicked.connect(self.edit_query)
        layout.addWidget(edit_btn)
        
        button_layout = QHBoxLayout()
        new_conv_btn = QPushButton("New Conversation")
        new_conv_btn.clicked.connect(self.new_conversation)
        save_btn = QPushButton("Save SQL")
        save_btn.clicked.connect(self.save_sql)
        load_btn = QPushButton("Load SQL File")
        load_btn.clicked.connect(self.load_sql)
        create_layer_btn = QPushButton("Create Layer")
        create_layer_btn.clicked.connect(self.create_layer)
        button_layout.addWidget(new_conv_btn)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(load_btn)
        button_layout.addWidget(create_layer_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_input.setText(file_path)
    
    def generate_query(self):
        image_path = self.image_input.text()
        prompt = self.prompt_input.toPlainText()
        
        if not image_path or not prompt:
            QMessageBox.warning(self, "Warning", "Please provide both image and prompt")
            return
        
        try:
            api_key = "..."
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel("gemini-2.5-flash")
            image = Image.open(image_path)
            
            full_prompt = f"{prompt}\nReturn ONLY the SQL query."
            response = model.generate_content([full_prompt, image])
            sql = response.text.strip()
            
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            self.sql_output.setText(sql)
            QMessageBox.information(self, "Success", "SQL query generated successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate query: {str(e)}")
    
    def save_sql(self):
        sql = self.sql_output.toPlainText()
        if not sql:
            QMessageBox.warning(self, "Warning", "No SQL query to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save SQL File", "", "SQL Files (*.sql)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(sql)
            QMessageBox.information(self, "Success", "SQL file saved successfully")
    
    def load_sql(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load SQL File", "", "SQL Files (*.sql)")
        if file_path:
            with open(file_path, 'r') as f:
                sql = f.read()
            self.sql_output.setText(sql)
    
    def edit_query(self):
        if self.sql_output.isReadOnly():
            self.sql_output.setReadOnly(False)
            self.sql_output.setStyleSheet("background-color: #ffffcc;")
            QMessageBox.information(self, "Edit Mode", "You can now edit the query")
        else:
            self.sql_output.setReadOnly(True)
            self.sql_output.setStyleSheet("")
            QMessageBox.information(self, "Read-Only Mode", "Query is now locked")
    
    def new_conversation(self):
        self.image_input.clear()
        self.prompt_input.clear()
        self.sql_output.clear()
        self.sql_output.setReadOnly(True)
        self.sql_output.setStyleSheet("")
    
    def create_layer(self):
        sql_query = self.sql_output.toPlainText()
        if not sql_query:
            QMessageBox.warning(self, "Warning", "No SQL query provided")
            return
        
        try:
            subquery = sql_query.strip()
            if subquery.endswith(';'):
                subquery = subquery[:-1].strip()
            
            uri = QgsDataSourceUri()
            uri.setConnection("127.0.0.1", "5432", "assignment1", "ismael", "ismael")
            uri.setDataSource("", f"({subquery})", "geom", "", "id")
            
            layer = QgsVectorLayer(uri.uri(), "SQL_Layer", "postgres")
            
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                iface.messageBar().pushMessage("Success", "Layer added successfully", level=0)
                QMessageBox.information(self, "Success", "Layer created successfully")
            else:
                error_msg = layer.error().message()
                iface.messageBar().pushMessage("Error", f"Failed to load layer: {error_msg}", level=2)
                QMessageBox.critical(self, "Error", f"Failed to create layer:\n{error_msg}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create layer: {str(e)}")

class SQLQueryGeneratorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        
    def initGui(self):
        self.action = QAction("SQL Query Generator", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&SQL Query Generator", self.action)
        
    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginMenu("&SQL Query Generator", self.action)
        
    def run(self):
        if self.dialog is None:
            self.dialog = SQLQueryGeneratorDialog()
        self.dialog.show()

def classFactory(iface):
    return SQLQueryGeneratorPlugin(iface)
