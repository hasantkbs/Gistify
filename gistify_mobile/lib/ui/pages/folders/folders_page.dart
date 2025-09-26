import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import 'folder_detail_page.dart';

class FoldersPage extends StatefulWidget {
  const FoldersPage({super.key});

  @override
  State<FoldersPage> createState() => _FoldersPageState();
}

class _FoldersPageState extends State<FoldersPage> {
  final List<_Folder> _folders = [];
  int _nextFolderIndex = 1;
  late final TextEditingController _folderNameController;

  static const List<Color> _palette = [
    AppColors.primaryBlue,
    Color(0xFF306BFF),
    Color(0xFF22C55E),
    Color(0xFF8B5CF6),
    Color(0xFFF97316),
    Color(0xFF14B8A6),
    Color(0xFFEF4444),
  ];

  @override
  void initState() {
    super.initState();
    _folderNameController = TextEditingController();
  }

  @override
  void dispose() {
    _folderNameController.dispose();
    super.dispose();
  }

  void _addFolder(String name) {
    setState(() {
      final color = _palette[(_nextFolderIndex - 1) % _palette.length];
      _folders.insert(
        0,
        _Folder(id: _nextFolderIndex, name: name, noteCount: 0, color: color),
      );
      _nextFolderIndex++;
    });
  }

  Future<void> _onCreateFolder() async {
    _folderNameController.text = 'New Folder $_nextFolderIndex';

    final result = await showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Create folder'),
          content: TextField(
            controller: _folderNameController,
            autofocus: true,
            decoration: const InputDecoration(hintText: 'Folder name'),
            onSubmitted: (value) => Navigator.pop(context, value),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () =>
                  Navigator.pop(context, _folderNameController.text),
              child: const Text('Create'),
            ),
          ],
        );
      },
    );

    if (!mounted) return;

    final name = result?.trim();
    if (name != null && name.isNotEmpty) {
      await Future<void>.delayed(Duration.zero);
      if (!mounted) return;
      _addFolder(name);
    }
    _folderNameController.clear();
  }

  Future<void> _renameFolder(_Folder folder) async {
    final controller = TextEditingController(text: folder.name);

    try {
      final result = await showDialog<String>(
        context: context,
        builder: (context) {
          return AlertDialog(
            title: const Text('Rename folder'),
            content: TextField(
              controller: controller,
              autofocus: true,
              decoration: const InputDecoration(hintText: 'Folder name'),
              onSubmitted: (value) => Navigator.pop(context, value),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(context, controller.text),
                child: const Text('Save'),
              ),
            ],
          );
        },
      );

      if (!mounted) return;

      final name = result?.trim();
      if (name != null && name.isNotEmpty && name != folder.name) {
        setState(() {
          final index = _folders.indexWhere(
            (element) => element.id == folder.id,
          );
          if (index != -1) {
            _folders[index] = _folders[index].copyWith(name: name);
          }
        });
      }
    } finally {
      controller.dispose();
    }
  }

  Future<void> _deleteFolder(_Folder folder) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Delete folder'),
          content: Text(
            "Delete '${folder.name}'? This action cannot be undone.",
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(context, true),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
              ),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );

    if (!mounted || confirmed != true) return;

    setState(() {
      _folders.removeWhere((element) => element.id == folder.id);
    });
  }

  void _openFolder(_Folder folder) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => FolderDetailPage(
          folderName: folder.name,
          noteCount: folder.noteCount,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 8),
              child: Row(
                children: [
                  const Text(
                    'Folders',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.3,
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    onPressed: _onCreateFolder,
                    icon: Container(
                      width: 40,
                      height: 40,
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppColors.primaryBlue,
                      ),
                      child: const Icon(Icons.add, color: Colors.white),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: _folders.isEmpty
                  ? const _EmptyState()
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      itemCount: _folders.length,
                      itemBuilder: (context, index) {
                        final folder = _folders[index];
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: ListTile(
                            onTap: () => _openFolder(folder),
                            tileColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(18),
                            ),
                            leading: Container(
                              width: 48,
                              height: 48,
                              decoration: BoxDecoration(
                                color: folder.color.withValues(alpha: 0.15),
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: Icon(Icons.folder, color: folder.color),
                            ),
                            title: Text(
                              folder.name,
                              style: const TextStyle(
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            subtitle: Text(
                              '${folder.noteCount} notes',
                              style: const TextStyle(
                                color: AppColors.textSecondary,
                              ),
                            ),
                            trailing: PopupMenuButton<_FolderMenuAction>(
                              onSelected: (action) {
                                switch (action) {
                                  case _FolderMenuAction.rename:
                                    _renameFolder(folder);
                                    break;
                                  case _FolderMenuAction.delete:
                                    _deleteFolder(folder);
                                    break;
                                }
                              },
                              itemBuilder: (context) => const [
                                PopupMenuItem(
                                  value: _FolderMenuAction.rename,
                                  child: Text('Rename'),
                                ),
                                PopupMenuItem(
                                  value: _FolderMenuAction.delete,
                                  child: Text('Delete'),
                                ),
                              ],
                              icon: const Icon(Icons.more_horiz),
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: const [
          Icon(Icons.folder_open, color: AppColors.textSecondary, size: 48),
          SizedBox(height: 12),
          Text(
            'No folders yet',
            style: TextStyle(
              color: AppColors.textSecondary,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

enum _FolderMenuAction { rename, delete }

class _Folder {
  const _Folder({
    required this.id,
    required this.name,
    required this.noteCount,
    required this.color,
  });

  final int id;
  final String name;
  final int noteCount;
  final Color color;

  _Folder copyWith({String? name, int? noteCount, Color? color}) {
    return _Folder(
      id: id,
      name: name ?? this.name,
      noteCount: noteCount ?? this.noteCount,
      color: color ?? this.color,
    );
  }
}
