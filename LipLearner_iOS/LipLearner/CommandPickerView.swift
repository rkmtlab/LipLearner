import Foundation
import UIKit
class CommandPickerView: UIPickerView, UIPickerViewDataSource, UIPickerViewDelegate {
    
    var registeredCommands: [String]!
    var textField: UITextField!
    init(registeredCommands: [String],
         frame: CGRect,
         textField: UITextField){
        super.init(frame: frame)
        self.registeredCommands = registeredCommands
        self.textField = textField
        self.textField.inputView = self
        let toolbar = UIToolbar()
        toolbar.frame = CGRect(x: 0, y: 0, width: 414, height: 44)
//        let doneButtonItem = UIBarButtonItem(barButtonSystemItem: .edit, target: self, action: #selector(CommandPickerView.donePicker))
        
        let textLabel = UILabel()
        textLabel.font = UIFont.systemFont(ofSize: 17)
        textLabel.text = "Select from existing commands" // Change this to be any string you want
        let textButton = UIBarButtonItem(customView: textLabel)
        let spacer = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        toolbar.setItems([spacer, textButton, spacer], animated: true)
        self.textField.inputAccessoryView = toolbar
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        1
    }

    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return registeredCommands.count
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return registeredCommands[row]
    }
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        textField.text = registeredCommands[row]
    }
    
//    @objc func donePicker() {
//        textField.endEditing(true)
//        textField.inputView = nil
//        self.textField.inputAccessoryView = nil
//        textField.becomeFirstResponder()
//    }
}


