import UIKit
import AVFoundation

protocol ItemSelectionViewControllerDelegate: AnyObject {
    func itemSelectionViewController(_ itemSelectionViewController: ItemSelectionViewController, identifier: String, reviewResults: [String?],
                                     didFinishSelectingRow selectedRow: Int)
}

struct conditionSelection {
    var count: Int
    var condition: [String]
}

class ItemSelectionViewController: UITableViewController {
    
    weak var delegate: ItemSelectionViewControllerDelegate?
    
    let identifier: String
    
    
    private let itemCellIdentifier = "Item"
    
    private var sections: [String]
    private var commandSet: [String]
    private var conditions: [String]
    private var recentCommands: [String]
    private var reviewResults: [String?]
    private var selectedRow : Int
    private var gifFiles = [URL]()
    init(delegate: ItemSelectionViewControllerDelegate,
         identifier: String,
         selectedRow: Int,
         recentCommands: [String],
         commandSet: [String]) {
        
        self.delegate = delegate
        self.identifier = identifier
        self.sections = ["Command registration langauge", "Review recent recognitions"]
        self.conditions = ["English", "日本語", "中文", "Melayu", "français", "español", "Tiếng Việt"]
        self.selectedRow = selectedRow
        self.recentCommands = recentCommands
        self.reviewResults = [String?](repeating: nil, count: recentCommands.count)
        self.commandSet = commandSet
        let documentsFolder = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let folderURL = documentsFolder.appendingPathComponent("gif")
        let folderExists = (try? folderURL.checkResourceIsReachable()) ?? false
        
        if folderExists {
            self.gifFiles = try! FileManager.default.contentsOfDirectory(at: folderURL,
                                                                         includingPropertiesForKeys:[.contentModificationDateKey],
                                                                         options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]).sorted(by: {let date0 = try $0.promisedItemResourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate!
                let date1 = try $1.promisedItemResourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate!
                return date0.compare(date1) == .orderedDescending
            })
        }
        super.init(style: .insetGrouped)
        navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(done))
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: itemCellIdentifier)
        
        view.tintColor = .black
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("`ItemSelectionViewController` cannot be initialized with `init(coder:)`")
    }
    
    
    @IBAction private func done() {
        // Notify the delegate that selecting items is finished.
        delegate?.itemSelectionViewController(self, identifier: self.identifier, reviewResults: self.reviewResults, didFinishSelectingRow: self.selectedRow)
        
        // Dismiss the view controller.
        dismiss(animated: true, completion: nil)
    }
    
    // MARK: UITableViewDataSource
    
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = tableView.dequeueReusableCell(withIdentifier: itemCellIdentifier, for: indexPath)
        var text: String!
        
        if indexPath.section == 0{
            text = conditions[indexPath.row]
            cell.accessoryType = selectedRow == indexPath.row ? .checkmark : .none
        }else{
            text = recentCommands[recentCommands.count - indexPath.row - 1]
            cell.accessoryType = .none
        }
        
        cell.textLabel?.text = text
        // Evaluate the semantic segmentation type to determine the label.
        cell.tintColor = UIColor.white
        
        return cell
    }
    
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        if section == 0{
            return self.conditions.count
        }else{
            return recentCommands.count
        }
    }
    
    // MARK: - UITableViewDelegate
    
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        if selectedRow != indexPath.row{ // unselected row
            if indexPath.section == 0 { // for language setting
                let itemToDeselect = IndexPath(row: selectedRow, section: indexPath.section) // deselect last selected row
                selectedRow = indexPath.row // register the new row
                tableView.reloadRows(at: [indexPath, itemToDeselect], with: .automatic) // reload the rows to be changed
            }
        }
        tableView.deselectRow(at: indexPath, animated: true)
        return
        
    }
    
    override func tableView(_ tableView: UITableView, titleForHeaderInSection section: Int) -> String? {
        return self.sections[section]
    }
    
    override func numberOfSections(in tableView: UITableView) -> Int {
        return self.sections.count
    }
    
    override func tableView(_ tableView: UITableView, trailingSwipeActionsConfigurationForRowAt indexPath: IndexPath) -> UISwipeActionsConfiguration? {
        if indexPath.section == 1{
            let editAction = UIContextualAction(style: .normal, title: "Edit") { (action, view, completionHandler) in
                let alert = UIAlertController(title: "Confirm", message: "Edit the label", preferredStyle: .alert)
                alert.addTextField { (textField) in
                    let textLabelText = self.tableView.cellForRow(at: indexPath)!.textLabel!.text
                    textField.text = textLabelText
                    var sortedCommands = self.commandSet
                    sortedCommands.removeAll(where: { $0 == textLabelText })
                    sortedCommands.insert(self.tableView.cellForRow(at: indexPath)!.textLabel!.text!, at: 0)
                    let pickerView = CommandPickerView(registeredCommands: sortedCommands, frame: CGRect(x: 5, y: 50, width: 250, height: 162), textField: textField)
                    pickerView.delegate = pickerView
                    pickerView.dataSource = pickerView
                    textField.inputView = pickerView
                    alert.view.addSubview(pickerView)
                    
                    if indexPath.row < self.gifFiles.count {
                        let gifData =  NSData(contentsOf: self.gifFiles[indexPath.row])
                        let animationGifView = UIWebView(frame: CGRect(x: 55 , y: 120, width: 160, height: 160))
                        //                            animationGifView.center = CGPoint(x:self.view.frame.width / 2.0,y:self.view.frame.height * 2 / 7.0)
                        animationGifView.load(gifData! as Data, mimeType: "image/gif", textEncodingName: "utf-8", baseURL: self.gifFiles[indexPath.row])
                        alert.view.addSubview(animationGifView)
                        let height = NSLayoutConstraint(item: alert.view!, attribute: .height, relatedBy: .equal, toItem: nil, attribute: .notAnAttribute, multiplier: 1, constant: 340)
                        alert.view.addConstraint(height)
                        alert.view.addSubview(animationGifView)
                    }
                }
                alert.addAction(UIAlertAction(title: "Confirm", style: .default, handler: { [alert] (_) in
                    let textFieldText = alert.textFields![0].text
                    self.tableView.cellForRow(at: indexPath)?.textLabel!.text = textFieldText
                    if alert.textFields![0].text != self.recentCommands[indexPath.row]{ // if the new command is different
                        self.reviewResults[indexPath.row] = textFieldText
                        self.tableView.cellForRow(at: indexPath)?.textLabel!.textColor = .systemCyan
                    }else{
                        self.tableView.cellForRow(at: indexPath)?.textLabel!.textColor = .white
                        self.reviewResults[indexPath.row] = nil
                    }
                }))
                alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: { (_) in }))
                self.present(alert, animated: true, completion:  nil)
                completionHandler(true)
            }
            editAction.backgroundColor = .systemBlue

            return UISwipeActionsConfiguration(actions: [editAction])
        }
        return nil
    }
}
