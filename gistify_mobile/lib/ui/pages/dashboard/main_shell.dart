import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../folders/folders_page.dart';
import '../notes/notes_page.dart';
import '../settings/settings_page.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key, this.initialIndex = notesTabIndex});

  static const route = '/home';
  static const notesTabIndex = 0;
  static const foldersTabIndex = 1;
  static const settingsTabIndex = 2;

  final int initialIndex;

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  late int _currentIndex;

  final _pages = const [
    NotesPage(),
    FoldersPage(),
    SettingsPage(),
  ];

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
  }

  @override
  void didUpdateWidget(covariant MainShell oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.initialIndex != widget.initialIndex) {
      _currentIndex = widget.initialIndex;
    }
  }

  void _onTap(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: _onTap,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        surfaceTintColor: Colors.white,
        height: 72,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.description_outlined),
            selectedIcon: Icon(Icons.description, color: AppColors.primaryBlue),
            label: 'Notes',
          ),
          NavigationDestination(
            icon: Icon(Icons.folder_outlined),
            selectedIcon: Icon(Icons.folder, color: AppColors.primaryBlue),
            label: 'Folders',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings, color: AppColors.primaryBlue),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
